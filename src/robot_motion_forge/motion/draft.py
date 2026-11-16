"""会话草稿：当前姿态、播放位置、选中帧与交互关节锁定。"""

from dataclasses import dataclass, field

import numpy as np

from robot_motion_forge.robot.model import FloatArray, Robot

from .project import Project


@dataclass
class Draft:
    qpos: FloatArray
    cursor: float = 0.0
    selected_key: str | None = None
    locked_joints: set[str] = field(default_factory=set[str])

    def validate(self, robot: Robot, project: Project) -> None:
        self.qpos = robot.validate_qpos(self.qpos)
        if not np.isfinite(self.cursor) or not 0 <= self.cursor <= project.duration:
            raise ValueError("播放位置必须在动作范围内")
        if self.locked_joints - set(robot.joint_names):
            raise ValueError("草稿锁定关节必须是模型中的可编辑关节")
        if self.selected_key is not None and not any(key.id == self.selected_key for key in project.keyframes):
            raise ValueError("选中帧必须是工程中的关键帧")

    def matches(self, other: "Draft") -> bool:
        return (
            np.array_equal(self.qpos, other.qpos)
            and self.cursor == other.cursor
            and self.selected_key == other.selected_key
            and self.locked_joints == other.locked_joints
        )
