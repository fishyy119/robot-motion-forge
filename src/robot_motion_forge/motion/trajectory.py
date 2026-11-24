"""轨迹数据、姿态插值与采样有效性检查。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Slerp

from robot_motion_forge.robot.closure import ClosureError
from robot_motion_forge.robot.model import JOINT_LIMIT_TOLERANCE, Robot
from robot_motion_forge.utils.array_types import FloatArray
from robot_motion_forge.utils.rotations import rotation_from_wxyz, rotation_to_wxyz

from .interpolation import RotationHermite, hermite_basis, segment_coordinates
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

    def _close_poses(self, robot: Robot, times: FloatArray, poses: FloatArray) -> None:
        closure = robot.closure
        if closure is None:
            return
        for index, time in enumerate(times):
            try:
                poses[index], _ = closure.reconstruct(poses[index])
            except ClosureError as exc:
                raise ClosureError(f"{time:.6g} s：{exc}") from exc


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
            # 延续首尾姿态，PCHIP 和 Log-Hermite 在两端的切线都为零。
            duration = last - first
            times = np.r_[first - duration, self.times, last + duration]
            return times, np.vstack((self.qpos[0], self.qpos, self.qpos[-1]))
        # 循环分量在首尾相同，两侧各补一个来自相邻周期的邻点。
        times = np.r_[first - (last - self.times[-2]), self.times, last + (self.times[1] - first)]
        return times, np.vstack((self.qpos[-2], self.qpos, self.qpos[1]))

    @cached_property
    def _linear_curves(self) -> dict[tuple[int, ...], PchipInterpolator]:
        return {}

    @cached_property
    def _xy_curve(self) -> PchipInterpolator:
        # 平移循环的水平位置只用本周期内的关键帧插值。
        return PchipInterpolator(self.times, self.qpos[:, :2], axis=0)

    def _sample_linear(self, robot: Robot, query: FloatArray, *, derivative: Literal[0, 1] = 0) -> FloatArray:
        indices = tuple(int(index) for index in robot.linear_qpos_indices)
        if not indices:
            return np.empty((len(query), 0))
        curve = self._linear_curves.get(indices)
        if curve is None:
            times, poses = self.interpolation_knots
            curve = PchipInterpolator(times, poses[:, indices], axis=0)
            self._linear_curves[indices] = curve
        result = curve(query, nu=derivative)
        if robot.floating and self.terminal_condition == "translation_loop":
            result[:, :2] = self._xy_curve(query, nu=derivative)
        return result

    def _sample_positions(self, robot: Robot, query: FloatArray) -> FloatArray:
        # 从动关节以默认姿态占位，按需由闭合关系重建。
        result = np.broadcast_to(robot.home, (len(query), robot.model.nq)).copy()
        result[:, robot.linear_qpos_indices] = self._sample_linear(robot, query)
        return result

    @cached_property
    def _rotation_curve(self) -> Slerp | RotationHermite:
        times, poses = self.interpolation_knots
        orientations = rotation_from_wxyz(poses[:, 3:7])
        if self.terminal_condition is not None:
            return RotationHermite(times, orientations)
        return Slerp(times, orientations)

    def sample(self, robot: Robot, times: ArrayLike) -> FloatArray:
        query = self._query_times(times)
        result = self._sample_positions(robot, query)
        if robot.floating:
            result[:, 3:7] = rotation_to_wxyz(self._rotation_curve(query))
        self._close_poses(robot, query, result)
        return result

    def sample_initial_state(self, robot: Robot, times: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """采样独立坐标的求解初值，按 MuJoCo 的 qpos 和 qvel 布局返回。"""
        query = self._query_times(times)
        qpos = self._sample_positions(robot, query)
        qvel = np.zeros((len(query), robot.model.nv))
        qvel[:, robot.linear_qvel_indices] = self._sample_linear(robot, query, derivative=1)
        if robot.floating:
            rotation_curve = self._rotation_curve
            if isinstance(rotation_curve, RotationHermite):
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
    def _rotation_curve(self) -> RotationHermite:
        rotations = rotation_from_wxyz(self.qpos[:, 3:7])
        omega = rotations.apply(self.qvel[:, 3:6])
        return RotationHermite(self.times, rotations, angular_velocities=omega)

    def _segment_coordinates(self, times: FloatArray) -> tuple[NDArray[np.intp], FloatArray, FloatArray]:
        left, step, u = segment_coordinates(self.times, times)
        return left, step[:, None], u[:, None]

    def sample(self, robot: Robot, times: ArrayLike) -> FloatArray:
        query = self._query_times(times)
        left, step, u = self._segment_coordinates(query)
        right = left + 1
        weights = hermite_basis(u, step)
        displacement, start_tangent, end_tangent = weights
        result = self.qpos[left].copy()
        qidx = robot.linear_qpos_indices
        vidx = robot.linear_qvel_indices
        result[:, qidx] += (
            displacement * (self.qpos[right][:, qidx] - result[:, qidx])
            + start_tangent * self.qvel[left][:, vidx]
            + end_tangent * self.qvel[right][:, vidx]
        )
        if robot.floating:
            result[:, 3:7] = rotation_to_wxyz(self._rotation_curve.sample_segments(left, weights))
        self._close_poses(robot, query, result)
        return result

    def sample_joint_velocities(self, robot: Robot, times: ArrayLike) -> FloatArray:
        """求关节位置曲线对时间的导数，列顺序为 robot.joint_names。"""
        left, step, u = self._segment_coordinates(self._query_times(times))
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
    velocities = trajectory.qvel if isinstance(trajectory, SolvedTrajectory) else None
    if velocities is not None and (
        velocities.shape != (len(times), robot.model.nv) or not np.isfinite(velocities).all()
    ):
        raise ValueError("轨迹 qvel 必须与模型维度一致且只包含有限数值")
    for index, (time, qpos) in enumerate(zip(times, trajectory.qpos, strict=True)):
        validated, lift = robot.validate_pose(qpos, jacobian=velocities is not None)
        if robot.closure is not None and np.linalg.norm(robot.difference(qpos, validated), ord=np.inf) > 1e-8:
            raise ClosureError(f"{time:.6g} s：轨迹节点的从动姿态与闭合重建不一致")
        if lift is not None and velocities is not None:
            velocity = velocities[index]
            if np.linalg.norm(lift @ velocity - velocity, ord=np.inf) > 1e-8:
                raise ClosureError(f"{time:.6g} s：轨迹节点的从动速度与闭合约束不一致")
    if velocities is None:
        return
    if robot.closure is not None:
        interior = times[:-1, None] + np.diff(times)[:, None] * (np.arange(1, 8) / 8)
        trajectory.sample(robot, interior.ravel())

    # 沿用求解器的 Bézier 控制点条件，验证三次 Hermite 曲线的段内关节位置。
    limited = np.flatnonzero(np.isfinite(robot.lower))
    positions = trajectory.qpos[:, robot.qadr[limited]]
    velocities = velocities[:, robot.vadr[limited]]
    step = np.diff(times)[:, None] / 3
    outgoing = positions[:-1] + step * velocities[:-1]
    incoming = positions[1:] - step * velocities[1:]
    lower = robot.lower[limited] - JOINT_LIMIT_TOLERANCE
    upper = robot.upper[limited] + JOINT_LIMIT_TOLERANCE
    outside = np.argwhere((np.minimum(outgoing, incoming) < lower) | (np.maximum(outgoing, incoming) > upper))
    if len(outside):
        segment, joint = outside[0]
        name = robot.joint_names[limited[joint]]
        raise ValueError(
            f"关节 {name} 在 {times[segment]:.6g}–{times[segment + 1]:.6g} s 区间的 Hermite 控制点超出位置限位"
        )
