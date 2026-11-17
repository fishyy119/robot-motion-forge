"""会话草稿：当前姿态、播放位置、选中帧与姿态编辑约束。"""

from dataclasses import dataclass, field
from typing import Self

import numpy as np

from robot_motion_forge.robot.model import FloatArray, Robot

from .constraints import EditConstraints
from .project import Project


@dataclass
class Draft:
    qpos: FloatArray
    cursor: float = 0.0
    selected_key: str | None = None
    constraints: EditConstraints = field(default_factory=EditConstraints)

    @classmethod
    def initial(cls, robot: Robot) -> Self:
        """按机器人配置建立初始姿态与默认部位固定状态。"""
        draft = cls(robot.home.copy())
        data = robot.forward(draft.qpos)
        for name, frame in robot.frames.items():
            if frame.fixed_mode is not None:
                draft.constraints.capture(robot, data, name, frame.fixed_mode)
        return draft

    def validate(self, robot: Robot, project: Project) -> None:
        self.qpos = robot.validate_qpos(self.qpos)
        if not np.isfinite(self.cursor) or not 0 <= self.cursor <= project.duration:
            raise ValueError("播放位置必须在动作范围内")
        self.constraints = EditConstraints.model_validate(self.constraints)
        self.constraints.validate_robot(robot)
        if (
            self.constraints.anchors
            and (name := self.constraints.violated_anchor(robot, robot.forward(self.qpos))) is not None
        ):
            raise ValueError(f"姿态超出部位固定容差：{name}")
        if self.selected_key is not None and not any(key.id == self.selected_key for key in project.keyframes):
            raise ValueError("选中帧必须是工程中的关键帧")

    def matches(self, other: "Draft") -> bool:
        return (
            np.array_equal(self.qpos, other.qpos)
            and self.cursor == other.cursor
            and self.selected_key == other.selected_key
            and self.constraints == other.constraints
        )
