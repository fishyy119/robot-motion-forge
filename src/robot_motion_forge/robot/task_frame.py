"""控制坐标系相对世界或指定连杆的位姿，以及相同参考系中的运动学雅可比。"""

from dataclasses import dataclass

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation

from robot_motion_forge.rotations import rotation_from_wxyz

from .model import FloatArray, Robot


@dataclass(frozen=True)
class FrameState:
    position: FloatArray
    orientation: Rotation
    # 前三行为线速度，后三行为参考坐标轴下的空间角速度；列对应 MuJoCo 的 nv。
    jacobian: FloatArray | None = None


class TaskFrame:
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
