"""机器人定义的 Pydantic 2 模型及 JSON Schema，不依赖本机资产。"""

import math
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

type Name = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
type Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]
type Vector3 = tuple[Number, Number, Number]
type Quaternion = tuple[Number, Number, Number, Number]
type JointRange = tuple[Number, Number]
type ControlMode = Literal["position", "orientation", "pose"]


class DefinitionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelConfig(DefinitionModel):
    name: Name = Field(description="机器人显示名称")
    repository_root: Name = Field(
        description="机器人仓库根目录；支持相对本定义的路径、~ 和环境变量；工程默认保存在该目录的 .robot-motion-forge 子目录"
    )
    model: Name = Field(
        description="本地纯机器人 MJCF 路径，不含地面等场景对象；相对机器人仓库根目录解析，也支持绝对路径、~ 和环境变量"
    )
    joint_limit_overrides: dict[Name, JointRange] = Field(
        default_factory=dict,
        description="覆盖模型关节限位，值为 [下限, 上限]；hinge 使用 rad，slide 使用 m；未列出的关节沿用模型限位",
    )

    @field_validator("joint_limit_overrides")
    @classmethod
    def validate_joint_limit_overrides(cls, value: dict[str, JointRange]) -> dict[str, JointRange]:
        for name, (lower, upper) in value.items():
            if lower >= upper:
                raise ValueError(f"关节 {name} 的限位下限必须小于上限")
        return value


class HomePose(DefinitionModel):
    position: Vector3 | None = Field(
        default=None, alias="position_m", description="torso 初始位置，单位 m；省略时使用模型 qpos0"
    )
    quaternion: Quaternion | None = Field(
        default=None, alias="quaternion_wxyz", description="torso 初始朝向，wxyz；读入后归一化"
    )
    joints_rad: dict[Name, Number] = Field(default_factory=dict, description="hinge 关节初始角度，单位 rad")
    joints_m: dict[Name, Number] = Field(default_factory=dict, description="slide 关节初始位移，单位 m")

    @field_validator("quaternion")
    @classmethod
    def normalize_quaternion(cls, value: Quaternion | None) -> Quaternion | None:
        if value is None:
            return None
        norm = math.hypot(*value)
        if not math.isfinite(norm) or norm < 1e-9:
            raise ValueError("home.quaternion_wxyz 必须是非零有限 wxyz 四元数")
        return value[0] / norm, value[1] / norm, value[2] / norm, value[3] / norm


class ControlFrame(DefinitionModel):
    """固定在连杆上的控制坐标系；原点可偏移，坐标轴与连杆对齐。"""

    body: Name = Field(description="控制坐标系所附着的 MuJoCo 连杆名称")
    offset: Vector3 = Field(
        default=(0.0, 0.0, 0.0), alias="offset_m", description="控制坐标系原点在连杆局部坐标中的位置，单位 m"
    )
    support_polygon_cm: list[Vector3] = Field(
        default_factory=list[Vector3],
        description="足底接触轮廓顶点，控制坐标系局部坐标，单位 cm；至少三个共面且不共线的点",
    )
    control_mode: ControlMode = Field(
        default="position", description="默认交互目标：位置、朝向或完整位姿；省略时为位置；可在界面中切换"
    )
    fixed_mode: ControlMode | None = Field(
        default=None, description="新会话的默认固定分量：位置、朝向或完整位姿；省略时不固定"
    )
    trajectory_mode: ControlMode | None = Field(
        default=None, description="新工程的默认轨迹求解分量：位置、朝向或完整位姿；省略时禁用"
    )

    @field_validator("support_polygon_cm")
    @classmethod
    def validate_support_polygon(cls, value: list[Vector3]) -> list[Vector3]:
        if not value:
            return value
        if len(value) < 3 or len(set(value)) != len(value):
            raise ValueError("足底轮廓至少需要三个不同的顶点")
        centered = np.asarray(value, dtype=float) - value[0]
        _, singular, axes = np.linalg.svd(centered, full_matrices=False)
        if singular[1] < 1e-8 or np.max(np.abs(centered @ axes[-1])) > 1e-6:
            raise ValueError("足底轮廓顶点必须共面且不共线")
        return value


class RobotDefinition(DefinitionModel):
    schema_url: str | None = Field(default=None, alias="$schema", description="可选的编辑器 JSON Schema 引用")
    robot: ModelConfig
    home: HomePose = Field(default_factory=HomePose)
    frames: dict[Name, ControlFrame] = Field(default_factory=dict, description="逻辑名称到控制坐标系定义的映射")
    joint_groups: dict[Name, list[Name]] = Field(
        default_factory=dict, description="界面中的关节分组；每个关节只能属于一个分组"
    )

    @field_validator("joint_groups")
    @classmethod
    def validate_joint_groups(cls, value: dict[str, list[str]]) -> dict[str, list[str]]:
        assigned: set[str] = set()
        for names in value.values():
            for name in names:
                if name in assigned:
                    raise ValueError(f"关节只能属于一个分组，且不可重复：{name}")
                assigned.add(name)
        return value

    @field_validator("frames")
    @classmethod
    def validate_frame_names(cls, value: dict[str, ControlFrame]) -> dict[str, ControlFrame]:
        for name in value:
            if "/" in name or name.startswith("joint:"):
                raise ValueError(f"坐标系名称不可包含 / 或以 joint: 开头：{name}")
        return value


def robot_definition_schema() -> dict[str, object]:
    schema = RobotDefinition.model_json_schema()
    schema.update(allowComments=True, allowTrailingCommas=True)
    return schema


if __name__ == "__main__":
    from pathlib import Path

    from robot_motion_forge.io.atomic import atomic_json

    path = Path("configs/robots/robot-definition.schema.json")
    atomic_json(path, robot_definition_schema())
    print(f"已导出机器人定义 Schema：{path}")
