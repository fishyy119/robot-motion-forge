"""姿态编辑目标与约束：关节锁定和控制坐标系的世界位姿锚定。"""

from dataclasses import dataclass
from typing import cast

import mujoco as mj
import numpy as np
from pydantic import Field, field_validator

from robot_motion_forge.robot.definition import ControlMode
from robot_motion_forge.robot.model import Robot
from robot_motion_forge.utils.array_types import FloatArray
from robot_motion_forge.utils.pydantic_models import Name
from robot_motion_forge.utils.rotations import normalize_wxyz, rotation_from_wxyz

from .types import JointNames, MotionModel

ANCHOR_POSITION_TOLERANCE_M = 1e-5
ANCHOR_ORIENTATION_TOLERANCE_RAD = 1e-4


@dataclass
class PoseTarget:
    """同一控制坐标系的世界位姿目标，控制模式决定参与求解的分量。"""

    frame: str
    position: list[float]
    quaternion: list[float] | None = None
    # 朝向任务仍保留位置供控件定位，但不将它计入求解。
    track_position: bool = True

    def validate(self, robot: Robot) -> None:
        robot.frame(self.frame)
        if np.shape(self.position) != (3,) or not np.isfinite(self.position).all():
            raise ValueError("位姿目标位置必须为三个有限数值")
        if not isinstance(cast(object, self.track_position), bool):
            raise ValueError("track_position 必须为布尔值")
        if not self.track_position and self.quaternion is None:
            raise ValueError("关闭位置跟踪时必须提供目标朝向")
        if self.quaternion is not None:
            try:
                self.quaternion = normalize_wxyz(self.quaternion).tolist()
            except ValueError as exc:
                raise ValueError("位姿目标方向必须为非零有限 wxyz 四元数") from exc


class FrameAnchor(MotionModel):
    mode: ControlMode
    position: list[float] = Field(min_length=3, max_length=3)
    quaternion: list[float] = Field(min_length=4, max_length=4)

    @field_validator("quaternion")
    @classmethod
    def normalize_quaternion(cls, value: list[float]) -> list[float]:
        # 草稿提交会递归重验；已归一化的世界锚点保持原值，避免舍入变化重置连续操作。
        try:
            return normalize_wxyz(value, preserve_unit=True).tolist()
        except ValueError as exc:
            raise ValueError("固定朝向必须为非零有限 wxyz 四元数") from exc

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
        position, orientation = robot.frame_pose(data, name)
        self.anchors[name] = FrameAnchor(mode=mode, position=position.tolist(), quaternion=orientation.tolist())

    def reanchor(self, robot: Robot, qpos: FloatArray) -> None:
        if self.anchors:
            data = robot.forward(qpos)
            for name, anchor in self.anchors.items():
                self.capture(robot, data, name, anchor.mode)

    def violated_anchor(self, robot: Robot, data: mj.MjData) -> str | None:
        for name, anchor in self.anchors.items():
            position, orientation = robot.frame_pose(data, name)
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
