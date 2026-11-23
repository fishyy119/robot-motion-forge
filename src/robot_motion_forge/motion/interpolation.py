"""分段插值、Hermite 基函数与 Log-Hermite 旋转曲线。"""

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from robot_motion_forge.utils.array_types import FloatArray
from robot_motion_forge.utils.rotations import exp_spatial_velocity, log_end_tangent

type HermiteWeights = tuple[FloatArray, FloatArray, FloatArray]


def hermite_basis(u: FloatArray, step: float | FloatArray, *, derivative: Literal[0, 1, 2] = 0) -> HermiteWeights:
    """对应位移、起点速度、终点速度的基函数；导数按实际时间计算。"""
    if derivative == 1:
        return 6 * u * (1 - u) / step, 3 * u**2 - 4 * u + 1, 3 * u**2 - 2 * u
    if derivative == 2:
        return 6 * (1 - 2 * u) / step**2, (6 * u - 4) / step, (6 * u - 2) / step
    return u**2 * (3 - 2 * u), step * u * (1 - u) ** 2, step * u**2 * (u - 1)


def segment_coordinates(knots: FloatArray, query: FloatArray) -> tuple[NDArray[np.intp], FloatArray, FloatArray]:
    """返回区间左端索引、区间时长和按区间时长归一化的查询位置。"""
    left = np.searchsorted(knots, query, side="right").clip(1, len(knots) - 1) - 1
    step = knots[left + 1] - knots[left]
    return left, step, (query - knots[left]) / step


def _estimate_angular_velocities(delta: FloatArray, steps: FloatArray) -> FloatArray:
    # 空间角速度用任务参考轴表达，相邻区间的转轴可直接比较。
    secants = delta / steps[:, None]
    speeds = np.linalg.norm(secants, axis=1)
    tangents = np.zeros((len(steps) + 1, 3))
    tangents[0], tangents[-1] = secants[0], secants[-1]
    for i in range(1, len(steps)):
        before, after = speeds[i - 1], speeds[i]
        if before == 0 or after == 0:
            continue
        # PCHIP 加权调和平均给出速率，两单位转轴的平均使折返减速到零。
        left, right = steps[i - 1], steps[i]
        w_left, w_right = left + 2 * right, 2 * left + right
        speed = (w_left + w_right) / (w_left / before + w_right / after)
        tangents[i] = 0.5 * speed * (secants[i - 1] / before + secants[i] / after)
    return tangents


class RotationHermite:
    """以每段起点为 Log 原点，使用共享的空间角速度做三次 Hermite 插值。"""

    def __init__(
        self, times: FloatArray, orientations: Rotation, *, angular_velocities: FloatArray | None = None
    ) -> None:
        self._times = times
        self._orientations = orientations
        self._delta = (orientations[1:] * orientations[:-1].inv()).as_rotvec()
        if angular_velocities is None:
            angular_velocities = _estimate_angular_velocities(self._delta, np.diff(times))
        self._start_tangent = angular_velocities[:-1]
        # Log 起点处的导数等于空间角速度，末端导数需乘以 J_l(delta)^(-1)。
        self._end_tangent = log_end_tangent(self._delta, angular_velocities[1:])

    def _log_vector(self, index: NDArray[np.intp], weights: HermiteWeights) -> FloatArray:
        displacement, start, end = weights
        return displacement * self._delta[index] + start * self._start_tangent[index] + end * self._end_tangent[index]

    def sample_segments(self, index: NDArray[np.intp], weights: HermiteWeights) -> Rotation:
        """使用已计算的区间索引与 Hermite 权重采样朝向。"""
        return Rotation.from_rotvec(self._log_vector(index, weights)) * self._orientations[index]

    def __call__(self, query: FloatArray) -> Rotation:
        index, step, fraction = segment_coordinates(self._times, query)
        return self.sample_segments(index, hermite_basis(fraction[:, None], step[:, None]))

    def sample_with_velocity(self, query: FloatArray) -> tuple[Rotation, FloatArray]:
        """同步求朝向和空间角速度，角速度相对于实际时间。"""
        index, step, fraction = segment_coordinates(self._times, query)
        u, h = fraction[:, None], step[:, None]
        vector = self._log_vector(index, hermite_basis(u, h))
        tangent = self._log_vector(index, hermite_basis(u, h, derivative=1))
        orientation = Rotation.from_rotvec(vector) * self._orientations[index]
        return orientation, exp_spatial_velocity(vector, tangent)
