"""轨迹数据、姿态插值与采样有效性检查。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation, Slerp

from robot_motion_forge.robot.model import Robot
from robot_motion_forge.utils.array_types import FloatArray
from robot_motion_forge.utils.rotations import log_end_tangent, rotation_from_wxyz, rotation_to_wxyz

from .interpolation import RotationBezier, hermite_basis, segment_coordinates
from .types import TerminalCondition


def _readonly(values: FloatArray) -> FloatArray:
    result = np.array(values, dtype=float, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class _TrajectoryNodes:
    times: FloatArray
    qpos: FloatArray

    def __post_init__(self) -> None:
        times = _readonly(self.times)
        if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError("轨迹必须包含至少两个严格递增的有限采样时刻")
        # 节点由轨迹独立持有且只读，插值器可在整个轨迹生命周期内复用。
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "qpos", _readonly(self.qpos))

    def _query_times(self, times: ArrayLike) -> FloatArray:
        return np.clip(np.atleast_1d(np.asarray(times, dtype=float)), self.times[0], self.times[-1])


@dataclass(frozen=True)
class ReferenceTrajectory(_TrajectoryNodes):
    terminal_condition: TerminalCondition | None = None

    @cached_property
    def interpolation_knots(self) -> tuple[FloatArray, FloatArray]:
        """用边界外的邻点让现有插值器计算循环切线与静止参考的首尾零切线。"""
        if self.terminal_condition is None:
            return self.times, self.qpos
        first, last = self.times[0], self.times[-1]
        if self.terminal_condition == "rest":
            # 延续首尾姿态，PCHIP 和球面 Bézier 在两端的切线都为零。
            duration = last - first
            times = np.r_[first - duration, self.times, last + duration]
            return times, np.vstack((self.qpos[0], self.qpos, self.qpos[-1]))
        # 循环分量在首尾相同，两侧各补一个来自相邻周期的邻点。
        times = np.r_[first - (last - self.times[-2]), self.times, last + (self.times[1] - first)]
        return times, np.vstack((self.qpos[-2], self.qpos, self.qpos[1]))

    @cached_property
    def _qpos_curve(self) -> PchipInterpolator:
        times, poses = self.interpolation_knots
        return PchipInterpolator(times, poses, axis=0)

    @cached_property
    def _xy_curve(self) -> PchipInterpolator:
        # 平移循环的水平位置只用本周期内的关键帧插值。
        return PchipInterpolator(self.times, self.qpos[:, :2], axis=0)

    def _sample_linear(self, robot: Robot, query: FloatArray, *, derivative: Literal[0, 1] = 0) -> FloatArray:
        result = self._qpos_curve(query, nu=derivative)
        if robot.floating and self.terminal_condition == "translation_loop":
            result[:, :2] = self._xy_curve(query, nu=derivative)
        return result

    @cached_property
    def _rotation_curve(self) -> Slerp | RotationBezier:
        times, poses = self.interpolation_knots
        orientations = rotation_from_wxyz(poses[:, 3:7])
        if self.terminal_condition is not None:
            return RotationBezier(times, orientations)
        return Slerp(times, orientations)

    def sample(self, robot: Robot, times: ArrayLike) -> FloatArray:
        query = self._query_times(times)
        result = self._sample_linear(robot, query)
        if robot.floating:
            result[:, 3:7] = rotation_to_wxyz(self._rotation_curve(query))
        return result

    def sample_state(self, robot: Robot, times: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """采样参考姿态及其时间导数，速度使用 MuJoCo 的广义坐标顺序。"""
        query = self._query_times(times)
        qpos = self._sample_linear(robot, query)
        linear_velocity = self._sample_linear(robot, query, derivative=1)
        qvel = np.empty((len(query), robot.model.nv))
        qvel[:, robot.linear_qvel_indices] = linear_velocity[:, robot.linear_qpos_indices]
        if robot.floating:
            rotation_curve = self._rotation_curve
            if isinstance(rotation_curve, RotationBezier):
                orientation, spatial_velocity = rotation_curve.sample_with_velocity(query)
                # MuJoCo 的基座角速度在基座局部坐标系中表达。
                angular_velocity = orientation.inv().apply(spatial_velocity)
            else:
                orientation = rotation_curve(query)
                # SLERP 的局部角速度在区间内恒定，内部节点取右侧导数。
                left, step, _ = segment_coordinates(self.times, query)
                a = rotation_from_wxyz(self.qpos[left, 3:7])
                b = rotation_from_wxyz(self.qpos[left + 1, 3:7])
                angular_velocity = (a.inv() * b).as_rotvec() / step[:, None]
            qpos[:, 3:7] = rotation_to_wxyz(orientation)
            qvel[:, 3:6] = angular_velocity
        return qpos, qvel


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
        left, step, u = segment_coordinates(self.times, self._query_times(times))
        return left, step[:, None], u[:, None]

    def sample(self, robot: Robot, times: ArrayLike) -> FloatArray:
        left, step, u = self._segment_coordinates(times)
        right = left + 1
        displacement, start_tangent, end_tangent = hermite_basis(u, step)
        result = self.qpos[left].copy()
        qidx = robot.linear_qpos_indices
        vidx = robot.linear_qvel_indices
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
