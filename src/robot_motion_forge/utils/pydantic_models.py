"""机器人定义、动作数据和工程文件共用的 Pydantic 校验规则。"""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints

type Name = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]


class StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False, validate_default=True)
