"""分段插值、Hermite 基函数与球面三次 Bézier 曲线。"""

from itertools import pairwise
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from robot_motion_forge.utils.array_types import FloatArray
from robot_motion_forge.utils.rotations import exp_spatial_velocity, log_end_tangent


def hermite_basis(
    u: FloatArray, step: float | FloatArray, derivative: Literal[0, 1, 2] = 0
) -> tuple[FloatArray, FloatArray, FloatArray]:
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


def _slerp(start: Rotation, end: Rotation, fraction: FloatArray) -> Rotation:
    """逐对沿最短弧插值，各对使用各自的时间进度。"""
    return Rotation.from_rotvec(fraction[:, None] * (end * start.inv()).as_rotvec()) * start


def _slerp_with_velocity(
    start: tuple[Rotation, FloatArray], end: tuple[Rotation, FloatArray], fraction: FloatArray
) -> tuple[Rotation, FloatArray]:
    start_rotation, start_velocity = start
    end_rotation, end_velocity = end
    # R(u) = Exp(u Log(Q)) R_a，Q = R_b R_a^(-1)。两端也可能随 u 变化。
    relative = end_rotation * start_rotation.inv()
    delta = relative.as_rotvec()
    delta_rate = log_end_tangent(delta, end_velocity - relative.apply(start_velocity))
    offset = fraction[:, None] * delta
    advance = Rotation.from_rotvec(offset)
    velocity = exp_spatial_velocity(offset, delta + fraction[:, None] * delta_rate)
    return advance * start_rotation, velocity + advance.apply(start_velocity)


class RotationBezier:
    """使用共享角速度连接相邻姿态的球面三次 Bézier 曲线。"""

    def __init__(self, times: FloatArray, orientations: Rotation) -> None:
        self.times = times
        self.orientations = orientations
        dt = np.diff(times)
        # 空间角速度用任务参考轴表达，相邻区间的转轴可直接比较。
        secants = (orientations[1:] * orientations[:-1].inv()).as_rotvec() / dt[:, None]
        speeds = np.linalg.norm(secants, axis=1)
        tangents = np.zeros((len(times), 3))
        tangents[0], tangents[-1] = secants[0], secants[-1]
        for i in range(1, len(times) - 1):
            before, after = speeds[i - 1], speeds[i]
            if before == 0 or after == 0:
                continue
            # PCHIP 加权调和平均给出速率，两单位转轴的平均使折返减速到零。
            left, right = dt[i - 1], dt[i]
            w_left, w_right = left + 2 * right, 2 * left + right
            speed = (w_left + w_right) / (w_left / before + w_right / after)
            tangents[i] = 0.5 * speed * (secants[i - 1] / before + secants[i] / after)

        # 三次 Bézier 的端点角速度是控制弧旋转向量的 3 / dt 倍。
        self.outgoing = Rotation.from_rotvec(dt[:, None] * tangents[:-1] / 3) * orientations[:-1]
        self.incoming = Rotation.from_rotvec(-dt[:, None] * tangents[1:] / 3) * orientations[1:]

    def _controls(self, index: NDArray[np.intp]) -> list[Rotation]:
        return [self.orientations[index], self.outgoing[index], self.incoming[index], self.orientations[index + 1]]

    def __call__(self, query: FloatArray) -> Rotation:
        index, _, fraction = segment_coordinates(self.times, query)
        controls = self._controls(index)
        # 球面 de Casteljau：四个控制朝向经过三层 SLERP 得到曲线上一点。
        for _ in range(3):
            controls = [_slerp(start, end, fraction) for start, end in pairwise(controls)]
        return controls[0]

    def sample_with_velocity(self, query: FloatArray) -> tuple[Rotation, FloatArray]:
        """同步求朝向和空间角速度，角速度相对于实际时间。"""
        index, step, fraction = segment_coordinates(self.times, query)
        zero = np.zeros((len(query), 3))
        states = [(control, zero) for control in self._controls(index)]
        # 对每层 SLERP 求关于 u 的导数，最后用 du/dt = 1 / step 换算。
        for _ in range(3):
            states = [_slerp_with_velocity(start, end, fraction) for start, end in pairwise(states)]
        orientation, velocity = states[0]
        return orientation, velocity / step[:, None]
