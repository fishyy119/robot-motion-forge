"""轨迹数据、姿态插值与采样有效性检查。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation, Slerp

from robot_motion_forge.robot.model import FloatArray, Robot
from robot_motion_forge.rotations import log_end_tangent, rotation_from_wxyz, rotation_to_wxyz

from .rotation_curve import RotationBezier
from .types import TerminalCondition


def hermite_basis(
    u: FloatArray, step: float | FloatArray, derivative: Literal[0, 1, 2] = 0
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """对应位移、起点速度、终点速度的基函数；导数按实际时间计算。"""
    if derivative == 1:
        return 6 * u * (1 - u) / step, 3 * u**2 - 4 * u + 1, 3 * u**2 - 2 * u
    if derivative == 2:
        return 6 * (1 - 2 * u) / step**2, (6 * u - 4) / step, (6 * u - 2) / step
    return u**2 * (3 - 2 * u), step * u * (1 - u) ** 2, step * u**2 * (u - 1)


def _readonly(values: FloatArray) -> FloatArray:
    result = np.array(values, dtype=float, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class _TrajectoryNodes:
    times: FloatArray
    qpos: FloatArray

    def __post_init__(self) -> None:
        # 节点由轨迹独立持有且只读，插值器可在整个轨迹生命周期内复用。
        object.__setattr__(self, "times", _readonly(self.times))
        object.__setattr__(self, "qpos", _readonly(self.qpos))

    def _query_times(self, times: ArrayLike) -> FloatArray:
        return np.clip(np.atleast_1d(np.asarray(times, dtype=float)), self.times[0], self.times[-1])


@dataclass(frozen=True)
class ReferenceTrajectory(_TrajectoryNodes):
    terminal_condition: TerminalCondition | None = None

    @cached_property
    def interpolation_knots(self) -> tuple[FloatArray, FloatArray]:
        """用边界外的邻点让现有插值器计算循环切线与静止参考的首尾零切线。"""
        if self.terminal_condition is None or len(self.times) < 2:
            return self.times, self.qpos
        first, last = self.times[0], self.times[-1]
        if self.terminal_condition == "rest":
            # 延续首尾姿态，PCHIP 和球面 Bézier 在两端的切线都为零。
            duration = last - first
            times = np.r_[first - duration, self.times, last + duration]
            return times, np.vstack((self.qpos[0], self.qpos, self.qpos[-1]))
        # 循环首尾是同一节点，两侧各补一个来自相邻周期的邻点。
        times = np.r_[first - (last - self.times[-2]), self.times, last + (self.times[1] - first)]
        return times, np.vstack((self.qpos[-2], self.qpos, self.qpos[1]))

    @cached_property
    def _qpos_curve(self) -> PchipInterpolator:
        times, poses = self.interpolation_knots
        return PchipInterpolator(times, poses, axis=0)

    @cached_property
    def _rotation_curve(self) -> Slerp | RotationBezier:
        times, poses = self.interpolation_knots
        orientations = rotation_from_wxyz(poses[:, 3:7])
        if self.terminal_condition is not None:
            return RotationBezier(times, orientations)
        return Slerp(times, orientations)

    def sample(self, robot: Robot, times: ArrayLike) -> FloatArray:
        query = self._query_times(times)
        if len(self.times) == 1:
            return np.repeat(self.qpos[:1], len(query), axis=0)
        result = self._qpos_curve(query)
        if robot.floating:
            result[:, 3:7] = rotation_to_wxyz(self._rotation_curve(query))
        return result


@dataclass(frozen=True)
class SolvedTrajectory(_TrajectoryNodes):
    # MuJoCo 广义速度，浮动基座的角速度使用基座局部坐标轴。
    qvel: FloatArray
    converged: bool
    message: str

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "qvel", _readonly(self.qvel))

    @cached_property
    def _angular_segments(self) -> tuple[Rotation, FloatArray, FloatArray, FloatArray]:
        rotations = rotation_from_wxyz(self.qpos[:, 3:7])
        delta = (rotations[1:] * rotations[:-1].inv()).as_rotvec()
        omega = rotations.apply(self.qvel[:, 3:6])
        return rotations[:-1], delta, omega[:-1], log_end_tangent(delta, omega[1:])

    def _segment_coordinates(self, times: ArrayLike) -> tuple[NDArray[np.intp], FloatArray, FloatArray]:
        query = self._query_times(times)
        left = np.clip(np.searchsorted(self.times, query, side="right") - 1, 0, len(self.times) - 2)
        step = (self.times[left + 1] - self.times[left])[:, None]
        u = (query - self.times[left])[:, None] / step
        return left, step, u

    def sample(self, robot: Robot, times: ArrayLike) -> FloatArray:
        left, step, u = self._segment_coordinates(times)
        right = left + 1
        displacement, start_tangent, end_tangent = hermite_basis(u, step)
        result = self.qpos[left].copy()
        qidx = np.r_[:3, robot.qadr] if robot.floating else robot.qadr
        vidx = np.r_[:3, robot.vadr] if robot.floating else robot.vadr
        result[:, qidx] += (
            displacement * (self.qpos[right][:, qidx] - result[:, qidx])
            + start_tangent * self.qvel[left][:, vidx]
            + end_tangent * self.qvel[right][:, vidx]
        )
        if robot.floating:
            rotations, delta, omega, end = self._angular_segments
            vector = displacement * delta[left] + start_tangent * omega[left] + end_tangent * end[left]
            result[:, 3:7] = rotation_to_wxyz(Rotation.from_rotvec(vector) * rotations[left])
        return result

    def sample_joint_velocities(self, robot: Robot, times: ArrayLike) -> FloatArray:
        """求关节位置曲线对时间的导数，列顺序为 robot.joint_names。"""
        left, step, u = self._segment_coordinates(times)
        right = left + 1
        displacement, start_tangent, end_tangent = hermite_basis(u, step, derivative=1)
        delta = self.qpos[right][:, robot.qadr] - self.qpos[left][:, robot.qadr]
        return (
            displacement * delta
            + start_tangent * self.qvel[left][:, robot.vadr]
            + end_tangent * self.qvel[right][:, robot.vadr]
        )


type Trajectory = ReferenceTrajectory | SolvedTrajectory


def validate_trajectory(robot: Robot, duration: float, trajectory: Trajectory) -> None:
    times = trajectory.times
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("轨迹必须包含至少两个严格递增的有限采样时刻")
    if abs(times[0]) > 1e-8 or abs(times[-1] - duration) > 1e-8:
        raise ValueError("轨迹必须覆盖完整动作时长")
    if trajectory.qpos.shape != (len(times), robot.model.nq):
        raise ValueError("轨迹 qpos 维度与模型不一致")
    if isinstance(trajectory, SolvedTrajectory) and (
        trajectory.qvel.shape != (len(times), robot.model.nv) or not np.isfinite(trajectory.qvel).all()
    ):
        raise ValueError("轨迹 qvel 必须与模型维度一致且只包含有限数值")
    for qpos in trajectory.qpos:
        robot.validate_qpos(qpos)
