"""受约束的增量拖拽：保持锁定关节，接受朝期望位姿的可行进展。"""

from collections.abc import Collection
from dataclasses import dataclass
from typing import Literal

import mink
import mujoco as mj
import numpy as np
from mink.limits.limit import Constraint
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation

from robot_motion_forge.motion.types import PoseTarget
from robot_motion_forge.robot.model import FloatArray, Robot, quaternion, rotation

from .ik import PoseIK, pose_error


@dataclass
class DragResult:
    qpos: FloatArray
    applied: bool
    position_error: float
    orientation_error: float
    reached: bool

    @property
    def state(self) -> Literal["reached", "projected", "blocked"]:
        if self.reached:
            return "reached"
        return "projected" if self.applied else "blocked"


class _IncrementLimit(mink.Limit):
    """包含浮动基座的单步分量边界；真实位移还需在积分后检查。"""

    def __init__(self, scales: FloatArray) -> None:
        self._matrix = np.vstack([np.eye(len(scales)), -np.eye(len(scales))])
        self._bounds = np.r_[scales, scales]

    def compute_qp_inequalities(self, configuration: mink.Configuration, dt: float) -> Constraint:
        return Constraint(G=self._matrix, h=self._bounds)


def _clamp(vector: FloatArray, maximum: float) -> FloatArray:
    return vector * min(1.0, maximum / max(np.linalg.norm(vector), 1e-12))


def follow_target(
    robot: Robot,
    qpos: ArrayLike,
    target: PoseTarget,
    *,
    lock_base: bool = True,
    locked_joints: Collection[str] = (),
    position_step: float = 0.003,
    rotation_step: float = np.pi / 60,
    position_tolerance: float = 0.001,
    orientation_tolerance: float = 0.03,
) -> DragResult:
    """执行一个有界增量；不可达时返回局部可行近似，受阻时保持原姿态。"""
    settings = [position_step, rotation_step, position_tolerance, orientation_tolerance]
    if not all(np.isfinite(value) and value > 0 for value in settings):
        raise ValueError("拖拽步长和误差容差必须是有限正数")
    position_tolerance = min(position_tolerance, position_step * 0.25)
    orientation_tolerance = min(orientation_tolerance, rotation_step * 0.25)
    qpos = robot.validate_qpos(qpos)
    qpos[robot.qadr] = np.clip(qpos[robot.qadr], robot.lower, robot.upper)
    m = robot.model
    # 整体归一化任务权重，保持位置 / 方向比例并改善 QP 数值尺度。
    normalization = max(1 / position_step, 0.25 / rotation_step)
    problem = PoseIK(
        robot,
        qpos,
        [target],
        lock_base=lock_base,
        locked_joints=locked_joints,
        position_cost=1 / (position_step * normalization),
        orientation_cost=0.25 / (rotation_step * normalization),
    )
    data = problem.configuration.data
    initial_error = pose_error(robot, data, target)

    def make_result(pose: FloatArray, applied: bool, error: FloatArray) -> DragResult:
        position = float(np.linalg.norm(error[:3]))
        orientation = float(np.linalg.norm(error[3:]))
        return DragResult(
            pose,
            applied,
            position,
            orientation,
            position <= position_tolerance and orientation <= orientation_tolerance,
        )

    frozen = set(problem.frozen.tolist())
    free_dofs = np.asarray([index for index in range(m.nv) if index not in frozen], dtype=int)
    initial = make_result(qpos, False, initial_error)
    if not len(free_dofs) or initial.reached:
        return initial

    initial_position, initial_orientation = robot.pose(data, target.end_effector)
    step_position = (
        initial_position + _clamp(np.asarray(target.position) - initial_position, position_step)
        if target.track_position
        else initial_position
    )
    step_orientation = initial_orientation
    if target.quaternion is not None:
        # 方向误差在世界坐标中限幅，再由 Mink 转换到任务坐标。
        angular_error = (rotation(target.quaternion) * rotation(initial_orientation).inv()).as_rotvec()
        step_orientation = quaternion(
            Rotation.from_rotvec(_clamp(angular_error, rotation_step)) * rotation(initial_orientation)
        )
    task = problem.tasks[0]
    assert isinstance(task, mink.FrameTask)
    task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(step_orientation), step_position))

    # 奇异位姿探测按单步范围归一化，使平移、旋转和不同尺寸机器人的探测尺度一致。
    scales = np.full(m.nv, rotation_step)
    if robot.floating:
        scales[:3] = position_step
    for joint in robot.joint_ids:
        if m.jnt_type[joint] == mj.mjtJoint.mjJNT_SLIDE:
            scales[m.jnt_dofadr[joint]] = position_step
    lower, upper = -scales, scales.copy()
    lower[robot.vadr] = np.maximum(lower[robot.vadr], robot.lower - qpos[robot.qadr])
    upper[robot.vadr] = np.minimum(upper[robot.vadr], robot.upper - qpos[robot.qadr])

    weights = np.r_[np.full(3, 1 / position_step), np.full(len(initial_error) - 3, 0.25 / rotation_step)]
    problem.limits.append(_IncrementLimit(scales))
    try:
        increment = problem.increment()
    except mink.NoSolutionFound:
        return initial
    if not np.isfinite(increment).all():
        return initial

    initial_cost = float(np.dot(weights * initial_error, weights * initial_error))
    candidate_data = mj.MjData(m)

    def advance(delta: FloatArray) -> DragResult | None:
        delta = np.clip(delta, lower, upper)
        delta[problem.frozen] = 0.0
        # 回溯后仍用非线性 FK 检查真实末端位移，避免线性近似超过单步范围。
        for fraction in (1.0, 0.5, 0.25, 0.125):
            increment = delta * fraction
            if robot.floating and (
                np.linalg.norm(increment[:3]) > position_step * 1.000001
                or np.linalg.norm(increment[3:6]) > rotation_step * 1.000001
            ):
                continue
            candidate = robot.integrate(qpos, increment)
            candidate[robot.qadr] = np.clip(candidate[robot.qadr], robot.lower, robot.upper)
            robot.forward(candidate, candidate_data)
            position, orientation = robot.pose(candidate_data, target.end_effector)
            if (target.track_position and np.linalg.norm(position - initial_position) > position_step * 1.000001) or (
                rotation(initial_orientation).inv() * rotation(orientation)
            ).magnitude() > rotation_step * 1.000001:
                continue
            error = pose_error(robot, candidate_data, target)
            value = float(np.dot(weights * error, weights * error))
            if value < initial_cost - max(1e-10, initial_cost * 1e-9):
                return make_result(candidate, True, error)
        return None

    candidate = advance(increment)
    if candidate is not None:
        return candidate

    # 近奇异姿态下一阶增量可能为零；只对影响目标的关节尝试受限探测。
    # 每个探测都必须降低实际误差并通过同样的限位与步长检查。
    jacobian = (
        np.asarray(task.cost[:, None] * task.compute_jacobian(problem.configuration))[:, free_dofs] * scales[free_dofs]
    )
    influential = np.linalg.norm(jacobian, axis=0) > 1e-7
    probe_dofs = free_dofs[influential]
    if not len(probe_dofs):
        return initial
    _, _, directions = np.linalg.svd(jacobian[:, influential], full_matrices=True)
    for direction in directions:
        for sign in (-1, 1):
            step = np.zeros(m.nv)
            step[probe_dofs] = sign * direction * scales[probe_dofs] / max(np.max(abs(direction)), 1e-12)
            candidate = advance(step)
            if candidate is not None:
                return candidate
    return initial
