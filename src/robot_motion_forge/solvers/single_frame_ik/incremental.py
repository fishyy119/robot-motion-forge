"""Mink / DAQP 单帧增量 IK，支持关节锁定与控制坐标系的世界位姿锚定。"""

# qpsolvers.solve_problem 尚未标注 **kwargs 的类型。
# pyright: reportUnknownMemberType=false

from dataclasses import dataclass

import mink
import mujoco as mj
import numpy as np
import qpsolvers

from robot_motion_forge.motion.pose_constraints import EditConstraints, PoseTarget
from robot_motion_forge.robot.model import Robot
from robot_motion_forge.utils.array_types import FloatArray

LINEAR_CONSTRAINT_TOLERANCE = 1e-9


class InconsistentConstraints(ValueError):
    """当前线性化中的硬约束没有共同解。"""


@dataclass(frozen=True)
class ConstraintSpace:
    # J Δq = b 的可行增量为 Δq = offset + basis z；basis 的列正交归一。
    offset: FloatArray
    basis: FloatArray

    @classmethod
    def from_equalities(cls, jacobian: FloatArray, feedback: FloatArray) -> "ConstraintSpace":
        size = jacobian.shape[1]
        if not len(jacobian):
            return cls(np.zeros(size), np.eye(size))
        left, singular, right = np.linalg.svd(jacobian, full_matrices=True)
        # 只消去机器精度下的相关行，保留模型中虽弱但真实存在的约束方向。
        threshold = np.finfo(float).eps * max(jacobian.shape) * singular[0] if len(singular) else 0.0
        rank = int(np.count_nonzero(singular > threshold))
        components = left[:, :rank].T @ feedback
        residual = feedback - left[:, :rank] @ components
        if np.linalg.norm(residual, ord=np.inf) > LINEAR_CONSTRAINT_TOLERANCE * max(
            1.0, float(np.linalg.norm(feedback, ord=np.inf))
        ):
            raise InconsistentConstraints("当前线性化中的固定约束不一致")
        offset = right[:rank].T @ (components / singular[:rank])
        return cls(offset, right[rank:].T)


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

    def __init__(self, robot: Robot, qpos: FloatArray, *, constraints: EditConstraints) -> None:
        self.robot = robot
        self.configuration = mink.Configuration(robot.model, qpos)
        posture_weights = np.zeros(robot.model.nv)
        posture_weights[robot.independent_dofs] = 0.001
        self._posture = mink.PostureTask(robot.model, cost=posture_weights)
        self._posture.set_target(qpos)
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

    def increment(self, target: mink.FrameTask | None = None) -> FloatArray:
        """求独立坐标增量，按 MuJoCo 的广义坐标顺序返回。"""
        # 虚拟时间步为 1，优化变量即本次切空间增量。
        self.configuration.check_limits(safety_break=False)
        tasks = [target, self._posture] if target is not None else [self._posture]
        damping = 1e-6
        closure = self.robot.closure
        problem = mink.build_ik(
            self.configuration, tasks, dt=1.0, damping=damping if closure is None else 0.0, limits=self.limits
        )
        matrix, feedback = self.equalities()
        chart = None
        if closure is not None:
            chart = self.robot.velocity_lift(self.configuration.q)[:, self.robot.independent_dofs]
            matrix = matrix @ chart
        try:
            space = ConstraintSpace.from_equalities(matrix, feedback)
        except InconsistentConstraints as exc:
            raise mink.NoSolutionFound("daqp") from exc
        origin, basis = space.offset, space.basis
        if chart is not None:
            origin, basis = chart @ origin, chart @ basis
        hessian = np.asarray(problem.P, dtype=float)
        gradient = np.asarray(problem.q, dtype=float)
        inequalities = np.asarray(problem.G, dtype=float) if problem.G is not None else None
        bounds = np.asarray(problem.h, dtype=float) if problem.h is not None else None
        if basis.shape[1]:
            # 在硬约束的零空间中优化软目标，近相关的等式不再进入 DAQP 的活动集。
            reduced_hessian = basis.T @ hessian @ basis
            if chart is not None:
                # 零空间基在独立坐标中正交，阻尼以独立坐标的增量为准。
                reduced_hessian += damping * np.eye(basis.shape[1])
            reduced = qpsolvers.Problem(
                reduced_hessian,
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
        if closure is not None:
            delta[closure.passive] = 0.0
        return delta
