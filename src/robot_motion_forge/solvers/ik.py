"""Mink / DAQP 多目标 IK，支持关节锁定与控制坐标系的世界位姿锚定。"""

# qpsolvers.solve_problem 尚未标注 **kwargs 的类型。
# pyright: reportUnknownMemberType=false

from dataclasses import dataclass

import mink
import mujoco as mj
import numpy as np
import qpsolvers
from numpy.typing import ArrayLike

from robot_motion_forge.motion.constraints import EditConstraints
from robot_motion_forge.motion.types import PoseTarget
from robot_motion_forge.robot.model import FloatArray, Robot
from robot_motion_forge.rotations import rotation_from_wxyz

from .constraint_space import LINEAR_CONSTRAINT_TOLERANCE, ConstraintSpace, InconsistentConstraints


def pose_error(robot: Robot, data: mj.MjData, target: PoseTarget) -> FloatArray:
    position, orientation = robot.pose(data, target.frame)
    # 保留位置、方向的三维分块，未启用的位置任务不产生误差。
    error = position - target.position if target.track_position else np.zeros(3)
    if target.quaternion is not None:
        error = np.r_[
            error, (rotation_from_wxyz(target.quaternion).inv() * rotation_from_wxyz(orientation)).as_rotvec()
        ]
    return error


class _AnchorTask(mink.Task):
    """选择锚定分量；位置直接约束世界原点，不耦合未锁定的朝向。"""

    def __init__(self, robot: Robot, target: PoseTarget) -> None:
        self.robot, self.target = robot, target
        self.site = robot.frame_site_ids[target.frame]
        self.orientation_task = mink.FrameTask(robot.model.site(self.site).name, "site", 0.0, 1.0)
        if target.quaternion is not None:
            self.orientation_task.set_target(
                mink.SE3.from_rotation_and_translation(
                    mink.SO3(np.asarray(target.quaternion)), np.asarray(target.position)
                )
            )
        super().__init__(cost=np.ones(3 * (int(target.track_position) + int(target.quaternion is not None))))

    def compute_error(self, configuration: mink.Configuration) -> FloatArray:
        error = np.empty(0)
        if self.target.track_position:
            error = configuration.data.site_xpos[self.site] - self.target.position
        if self.target.quaternion is not None:
            error = np.r_[error, self.orientation_task.compute_error(configuration)[3:]]
        return error

    def compute_jacobian(self, configuration: mink.Configuration) -> FloatArray:
        jacobian = np.empty((0, self.robot.model.nv))
        if self.target.track_position:
            jacobian = np.empty((3, self.robot.model.nv))
            mj.mj_jacSite(self.robot.model, configuration.data, jacobian, None, self.site)
        if self.target.quaternion is not None:
            jacobian = np.vstack((jacobian, self.orientation_task.compute_jacobian(configuration)[3:]))
        return jacobian


class PoseIK:
    """同一次姿态求解的任务、硬锁定与 MuJoCo 运动学状态。"""

    def __init__(
        self,
        robot: Robot,
        qpos: FloatArray,
        targets: list[PoseTarget],
        *,
        constraints: EditConstraints,
        position_cost: float = 1.0,
        orientation_cost: float = 0.3,
    ) -> None:
        self.configuration = mink.Configuration(robot.model, qpos)
        self.tasks: list[mink.Task] = []
        for target in targets:
            target.validate(robot)
            _, orientation = robot.pose(self.configuration.data, target.frame)
            task = mink.FrameTask(
                robot.model.site(robot.frame_site_ids[target.frame]).name,
                "site",
                position_cost=position_cost if target.track_position else 0.0,
                orientation_cost=orientation_cost if target.quaternion is not None else 0.0,
            )
            task.set_target(
                mink.SE3.from_rotation_and_translation(
                    mink.SO3(np.asarray(target.quaternion) if target.quaternion is not None else orientation),
                    np.asarray(target.position),
                )
            )
            self.tasks.append(task)
        posture = mink.PostureTask(robot.model, cost=0.001)
        posture.set_target(qpos)
        self.tasks.append(posture)
        constraints.validate_robot(robot)
        frozen = [int(robot.vadr[robot.joint_names.index(name)]) for name in constraints.locked_joints]
        self.frozen = np.asarray(sorted(set(frozen)), dtype=int)
        self.constraints: list[mink.Task] = (
            [mink.DofFreezingTask(robot.model, self.frozen.tolist())] if len(self.frozen) else []
        )
        self.constraints.extend(_AnchorTask(robot, anchor.target(name)) for name, anchor in constraints.anchors.items())
        self.limits: list[mink.Limit] = [mink.ConfigurationLimit(robot.model, gain=1.0)]

    def equalities(self) -> tuple[FloatArray, FloatArray]:
        if not self.constraints:
            return np.empty((0, self.configuration.model.nv)), np.empty(0)
        return (
            np.vstack([task.compute_jacobian(self.configuration) for task in self.constraints]),
            np.concatenate([-task.gain * task.compute_error(self.configuration) for task in self.constraints]),
        )

    def increment(self) -> FloatArray:
        # 虚拟时间步为 1，优化变量即本次切空间增量。
        self.configuration.check_limits(safety_break=False)
        problem = mink.build_ik(self.configuration, self.tasks, dt=1.0, damping=1e-6, limits=self.limits)
        try:
            space = ConstraintSpace.from_equalities(*self.equalities())
        except InconsistentConstraints as exc:
            raise mink.NoSolutionFound("daqp") from exc
        origin, basis = space.offset, space.basis
        hessian = np.asarray(problem.P, dtype=float)
        gradient = np.asarray(problem.q, dtype=float)
        inequalities = np.asarray(problem.G, dtype=float) if problem.G is not None else None
        bounds = np.asarray(problem.h, dtype=float) if problem.h is not None else None
        if basis.shape[1]:
            # 在硬约束的零空间中优化软目标，近相关的等式不再进入 DAQP 的活动集。
            reduced = qpsolvers.Problem(
                basis.T @ hessian @ basis,
                basis.T @ (hessian @ origin + gradient),
                inequalities @ basis if inequalities is not None else None,
                bounds - inequalities @ origin if bounds is not None and inequalities is not None else None,
            )
            result = qpsolvers.solve_problem(reduced, solver="daqp", primal_tol=LINEAR_CONSTRAINT_TOLERANCE)
            if not result.found or result.x is None:
                raise mink.NoSolutionFound("daqp")
            delta = origin + basis @ result.x
        else:
            delta = origin.copy()
        if (
            inequalities is not None
            and bounds is not None
            and np.any(inequalities @ delta > bounds + 10 * LINEAR_CONSTRAINT_TOLERANCE)
        ):
            raise mink.NoSolutionFound("daqp")
        # 将硬锁定的数值舍入误差归零，避免连续交互积累漂移。
        delta[self.frozen] = 0.0
        return delta


@dataclass
class IKResult:
    qpos: FloatArray
    success: bool
    position_error: float
    orientation_error: float


def solve_ik(
    robot: Robot,
    qpos: ArrayLike,
    targets: list[PoseTarget],
    *,
    constraints: EditConstraints | None = None,
    max_iterations: int = 60,
) -> IKResult:
    qpos = robot.validate_qpos(qpos)
    if not targets:
        return IKResult(qpos, True, 0.0, 0.0)
    constraints = constraints if constraints is not None else EditConstraints()
    problem = PoseIK(robot, qpos, targets, constraints=constraints)
    for _ in range(max_iterations):
        try:
            delta = problem.increment()
        except mink.NoSolutionFound:
            break
        if not np.isfinite(delta).all():
            break
        solved = robot.integrate(problem.configuration.q, delta)
        solved[robot.qadr] = np.clip(solved[robot.qadr], robot.lower, robot.upper)
        problem.configuration.update(solved)
        errors = [pose_error(robot, problem.configuration.data, target) for target in targets]
        if max(np.linalg.norm(error) for error in errors) < 1e-5 or np.linalg.norm(delta) < 1e-9:
            break
    errors = [pose_error(robot, problem.configuration.data, target) for target in targets]
    position = max(float(np.linalg.norm(error[:3])) for error in errors)
    orientation = max((float(np.linalg.norm(error[3:])) for error in errors if len(error) == 6), default=0.0)
    success = (
        position <= 0.01
        and orientation <= 0.08
        and constraints.violated_anchor(robot, problem.configuration.data) is None
    )
    return IKResult(problem.configuration.q.copy(), success, position, orientation)
