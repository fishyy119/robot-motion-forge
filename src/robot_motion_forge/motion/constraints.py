"""姿态编辑约束：关节锁定与控制坐标系的世界位姿锚定。"""

import math

import mujoco as mj
import numpy as np
from pydantic import Field, field_validator

from robot_motion_forge.robot.definition import ControlMode
from robot_motion_forge.robot.model import FloatArray, Robot
from robot_motion_forge.rotations import rotation_from_wxyz

from .types import JointNames, MotionModel, Name, PoseTarget

ANCHOR_POSITION_TOLERANCE_M = 1e-5
ANCHOR_ORIENTATION_TOLERANCE_RAD = 1e-4


class FrameAnchor(MotionModel):
    mode: ControlMode
    position: list[float] = Field(min_length=3, max_length=3)
    quaternion: list[float] = Field(min_length=4, max_length=4)

    @field_validator("quaternion")
    @classmethod
    def normalize_quaternion(cls, value: list[float]) -> list[float]:
        norm = math.hypot(*value)
        if not math.isfinite(norm) or norm < 1e-9:
            raise ValueError("固定朝向必须为非零有限 wxyz 四元数")
        # 草稿提交会递归重验；已归一化的世界锚点保持原值，避免舍入变化重置连续操作。
        if abs(norm - 1.0) <= 1e-12:
            return value
        return (np.asarray(value) / norm).tolist()

    def target(self, name: str) -> PoseTarget:
        return PoseTarget(
            name,
            self.position.copy(),
            self.quaternion.copy() if self.mode != "position" else None,
            track_position=self.mode != "orientation",
        )


class EditConstraints(MotionModel):
    locked_joints: JointNames = Field(default_factory=list[str])
    anchors: dict[Name, FrameAnchor] = Field(default_factory=dict[str, FrameAnchor])

    def validate_robot(self, robot: Robot) -> None:
        if set(self.locked_joints) - set(robot.joint_names):
            raise ValueError("锁定关节必须是模型中的可编辑关节")
        for name in self.anchors:
            robot.frame(name)

    def capture(self, robot: Robot, data: mj.MjData, name: str, mode: ControlMode) -> None:
        position, orientation = robot.pose(data, name)
        self.anchors[name] = FrameAnchor(mode=mode, position=position.tolist(), quaternion=orientation.tolist())

    def reanchor(self, robot: Robot, qpos: FloatArray) -> None:
        if self.anchors:
            data = robot.forward(qpos)
            for name, anchor in self.anchors.items():
                self.capture(robot, data, name, anchor.mode)

    def violated_anchor(self, robot: Robot, data: mj.MjData) -> str | None:
        for name, anchor in self.anchors.items():
            position, orientation = robot.pose(data, name)
            if (
                anchor.mode != "orientation"
                and np.linalg.norm(position - anchor.position) > ANCHOR_POSITION_TOLERANCE_M
            ):
                return name
            if (
                anchor.mode != "position"
                and (rotation_from_wxyz(anchor.quaternion).inv() * rotation_from_wxyz(orientation)).magnitude()
                > ANCHOR_ORIENTATION_TOLERANCE_RAD
            ):
                return name
        return None
