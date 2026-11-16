"""Mink / DAQP 多目标 IK，支持固定基座与指定关节。"""

# Mink 的 solve_ik 尚未标注 **kwargs 的类型。
# pyright: reportUnknownMemberType=false

from collections.abc import Collection
from dataclasses import dataclass

import mink
import mujoco as mj
import numpy as np
from numpy.typing import ArrayLike

from robot_motion_forge.motion.types import PoseTarget
from robot_motion_forge.robot.model import FloatArray, Robot, rotation


def pose_error(robot: Robot, data: mj.MjData, target: PoseTarget) -> FloatArray:
    position, orientation = robot.pose(data, target.end_effector)
    # 保留位置、方向的三维分块，未启用的位置任务不产生误差。
    error = position - target.position if target.track_position else np.zeros(3)
    if target.quaternion is not None:
        error = np.r_[error, (rotation(target.quaternion).inv() * rotation(orientation)).as_rotvec()]
    return error


class PoseIK:
    """同一次姿态求解的任务、硬锁定与 MuJoCo 运动学状态。"""

    def __init__(
        self,
        robot: Robot,
        qpos: FloatArray,
        targets: list[PoseTarget],
        *,
        lock_base: bool,
        locked_joints: Collection[str],
        position_cost: float = 1.0,
        orientation_cost: float = 0.3,
    ) -> None:
        self.configuration = mink.Configuration(robot.model, qpos)
        self.tasks: list[mink.Task] = []
        for target in targets:
            target.validate(robot)
            _, orientation = robot.pose(self.configuration.data, target.end_effector)
            task = mink.FrameTask(
                robot.end_effector_sites[target.end_effector],
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
        frozen = list(range(6)) if lock_base and robot.floating else []
        frozen.extend(int(robot.vadr[robot.joint_names.index(name)]) for name in locked_joints)
        self.frozen = np.asarray(sorted(set(frozen)), dtype=int)
        self.constraints: list[mink.Task] = (
            [mink.DofFreezingTask(robot.model, self.frozen.tolist())] if len(self.frozen) else []
        )
        self.limits: list[mink.Limit] = [mink.ConfigurationLimit(robot.model, gain=1.0)]

    def increment(self) -> FloatArray:
        # 虚拟时间步为 1，Mink 返回的速度数值即本次切空间增量。
        delta = np.asarray(
            mink.solve_ik(
                self.configuration,
                self.tasks,
                dt=1.0,
                solver="daqp",
                damping=1e-6,
                limits=self.limits,
                constraints=self.constraints,
            ),
            dtype=float,
        )
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
    lock_base: bool = True,
    max_iterations: int = 60,
    locked_joints: Collection[str] = (),
) -> IKResult:
    qpos = robot.validate_qpos(qpos)
    if not targets:
        return IKResult(qpos, True, 0.0, 0.0)
    problem = PoseIK(robot, qpos, targets, lock_base=lock_base, locked_joints=locked_joints)
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
    return IKResult(problem.configuration.q.copy(), position <= 0.01 and orientation <= 0.08, position, orientation)
