"""关键帧、位姿目标和求解参数的数据定义。"""

from dataclasses import dataclass
from typing import Annotated, Literal, cast
from uuid import uuid4

import numpy as np
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints

from robot_motion_forge.robot.definition import ControlMode
from robot_motion_forge.robot.model import Robot

type Name = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
type Weight = Annotated[float, Field(ge=0)]
type TerminalCondition = Literal["rest", "loop"]


def _unique_joint_names(names: list[str]) -> list[str]:
    if len(set(names)) != len(names):
        raise ValueError("关节名称不可重复")
    return names


type JointNames = Annotated[list[Name], AfterValidator(_unique_joint_names)]


class MotionModel(BaseModel):
    # 工程允许批量修改；提交时递归重验，避免中间态触发跨字段校验。
    model_config = ConfigDict(
        strict=True, extra="forbid", allow_inf_nan=False, validate_default=True, revalidate_instances="always"
    )


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
            q = np.asarray(self.quaternion, dtype=float)
            if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-9:
                raise ValueError("位姿目标方向必须为非零有限 wxyz 四元数")
            self.quaternion = (q / np.linalg.norm(q)).tolist()


class Keyframe(MotionModel):
    time: float = Field(ge=0)
    qpos: list[float]
    name: str = "关键帧"
    locked: bool = True
    zero_velocity: bool = False
    id: Name = Field(default_factory=lambda: uuid4().hex[:12])


class FrameTask(MotionModel):
    mode: ControlMode | None = None
    reference_body: Name | None
    weight: Weight = 1.0


class SolverSettings(MotionModel):
    terminal_condition: TerminalCondition = "rest"
    velocity: Weight = 0.002
    acceleration: Weight = 0.002
    keyframe: Weight = 200.0
    # 碰撞权重作用于按 1 cm 归一化的距离误差平方。
    self_collision: Weight = 1.0
    ground_collision: Weight = 1.0
    collision_margin_cm: float = Field(default=0.3, ge=0)
    foot_support: bool = False
    support_position: Weight = 1.0
    support_orientation: Weight = 1.0
    balance: Weight = 1.0
    support_margin_cm: float = Field(default=0.5, ge=0)
    frame_reference: Weight = 1.0
    frame_velocity: Weight = 0.002
    frame_acceleration: Weight = 0.002
    max_iterations: int = Field(default=60, ge=1)
    solve_fps: float = Field(default=15.0, gt=0)
