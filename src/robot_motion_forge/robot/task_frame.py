"""末端相对世界或指定连杆的位姿，以及相同参考系中的运动学雅可比。"""

from dataclasses import dataclass

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation

from .model import FloatArray, Robot, rotation


@dataclass(frozen=True)
class FrameState:
    position: FloatArray
    orientation: Rotation
    # 前三行为线速度，后三行为参考坐标轴下的空间角速度；列对应 MuJoCo 的 nv。
    jacobian: FloatArray | None = None


class TaskFrame:
    def __init__(self, robot: Robot, name: str, reference_body: str | None) -> None:
        self.robot, self.name = robot, name
        self.site = mj.mj_name2id(robot.model, mj.mjtObj.mjOBJ_SITE, robot.end_effector_sites[name])
        self.reference = robot.body_id(reference_body) if reference_body is not None else None
        self.data = mj.MjData(robot.model)

    def state(self, pose: FloatArray, *, jacobian: bool = False) -> FrameState:
        robot, data = self.robot, self.data
        robot.forward(pose, data)
        position, quat = robot.pose(data, self.name)
        orientation = rotation(quat)
        jp, jr = np.zeros((3, robot.model.nv)), np.zeros((3, robot.model.nv))
        if jacobian:
            mj.mj_jacSite(robot.model, data, jp, jr, self.site)
        if self.reference is not None:
            inverse = rotation(data.xquat[self.reference]).inv()
            offset = position - data.xpos[self.reference]
            position = inverse.apply(offset)
            orientation = inverse * orientation
            if jacobian:
                bp, br = np.empty_like(jp), np.empty_like(jr)
                mj.mj_jacBody(robot.model, data, bp, br, self.reference)
                # d[R_b^T(p_e-p_b)] = R_b^T(dp_e-dp_b-omega_b x (p_e-p_b))。
                jp = inverse.as_matrix() @ (jp - bp - np.cross(br.T, offset).T)
                jr = inverse.as_matrix() @ (jr - br)
        return FrameState(position, orientation, np.vstack((jp, jr), dtype=np.float64) if jacobian else None)
