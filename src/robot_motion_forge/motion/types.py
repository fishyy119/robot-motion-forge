"""动作数据共用的模型基类和类型约束。"""

from typing import Annotated, Literal

from pydantic import AfterValidator, ConfigDict

from robot_motion_forge.utils.pydantic_models import Name, StrictModel

type TerminalCondition = Literal["rest", "loop", "translation_loop"]


def _unique_joint_names(names: list[str]) -> list[str]:
    if len(set(names)) != len(names):
        raise ValueError("关节名称不可重复")
    return names


type JointNames = Annotated[list[Name], AfterValidator(_unique_joint_names)]


class MotionModel(StrictModel):
    # 工程允许批量修改；提交时递归重验，避免中间态触发跨字段校验。
    model_config = ConfigDict(revalidate_instances="always")
