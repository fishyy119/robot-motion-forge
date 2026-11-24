"""MuJoCo 正向运动学，以及控制坐标系的位姿和雅可比。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco as mj
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation

from robot_motion_forge.utils.array_types import FloatArray
from robot_motion_forge.utils.rotations import rotation_from_wxyz, skew

if TYPE_CHECKING:
    from .model import Robot


def forward_kinematics(model: mj.MjModel, qpos: ArrayLike, data: mj.MjData | None = None) -> mj.MjData:
    state = mj.MjData(model) if data is None else data
    state.qpos[:] = qpos
    mj.mj_kinematics(model, state)
    mj.mj_comPos(model, state)
    return state


@dataclass(frozen=True)
class FrameState:
    position: FloatArray
    orientation: Rotation
    # 前三行为线速度，后三行为参考坐标轴下的空间角速度；列对应 MuJoCo 的 nv。
    jacobian: FloatArray | None = None


class ControlFrameKinematics:
    def __init__(self, robot: Robot, name: str, reference_body: str | None) -> None:
        self.robot = robot
        self.site = robot.frame_site_ids[name]
        self.reference = robot.frame_reference_id(name, reference_body)
        self.data = mj.MjData(robot.model)

    def state(self, pose: FloatArray, *, jacobian: bool = False) -> FrameState:
        robot, data = self.robot, self.data
        robot.forward(pose, data)
        position = data.site_xpos[self.site].copy()
        orientation = Rotation.from_matrix(data.site_xmat[self.site].reshape(3, 3))
        derivative = np.empty((6, robot.model.nv)) if jacobian else None
        if derivative is not None:
            mj.mj_jacSite(robot.model, data, derivative[:3], derivative[3:], self.site)
        if self.reference is not None:
            inverse = rotation_from_wxyz(data.xquat[self.reference]).inv()
            offset = position - data.xpos[self.reference]
            position = inverse.apply(offset)
            orientation = inverse * orientation
            if derivative is not None:
                bp, br = np.empty((3, robot.model.nv)), np.empty((3, robot.model.nv))
                mj.mj_jacBody(robot.model, data, bp, br, self.reference)
                # d[R_b^T(p_e-p_b)] = R_b^T(dp_e-dp_b-omega_b x (p_e-p_b))。
                matrix = inverse.as_matrix()
                derivative[:3] = matrix @ (derivative[:3] - bp - np.cross(br.T, offset).T)
                derivative[3:] = matrix @ (derivative[3:] - br)
        return FrameState(position, orientation, derivative)

    def velocity_pose_jacobian(self, pose: FloatArray, velocity: FloatArray, jacobian: FloatArray) -> FloatArray:
        """在广义速度固定时，对姿态求 J(q)v 的导数。"""
        if not np.any(velocity):
            return np.zeros_like(jacobian)
        # 交换混合偏导，只沿 v 做一次中心差分。基座右增量与任务空间角速度
        # 的坐标轴随姿态转动，交换时分别加上输入和输出的李括号项。
        step = 1e-5 / max(1.0, float(np.linalg.norm(velocity, ord=np.inf)))
        offset = step * velocity
        robot = self.robot
        plus_pose = robot.integrate(pose, offset)
        minus_pose = robot.integrate(pose, -offset)
        plus_lift, minus_lift = None, None
        if robot.closure is not None:
            # 导数探测可越过限位，实际节点仍须满足从动关节限位。
            plus_pose, plus_lift = robot.closure.reconstruct(plus_pose, jacobian=True, limits=False)
            minus_pose, minus_lift = robot.closure.reconstruct(minus_pose, jacobian=True, limits=False)
        plus = self.state(plus_pose, jacobian=True).jacobian
        minus = self.state(minus_pose, jacobian=True).jacobian
        assert plus is not None and minus is not None
        if plus_lift is not None and minus_lift is not None:
            plus = plus @ plus_lift
            minus = minus @ minus_lift
        derivative = (plus - minus) / (2 * step)
        if robot.floating:
            derivative[:, 3:6] -= jacobian[:, 3:6] @ skew(velocity[3:6])
        derivative[3:] -= skew(jacobian[3:] @ velocity) @ jacobian[3:]
        return derivative
