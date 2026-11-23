"""通用路径解析。"""

import os
from pathlib import Path


def resolve_path(value: str | Path, *, base_dir: Path | None = None) -> Path:
    """展开字符串中的环境变量，按指定目录或当前工作目录解析路径。"""
    path = Path(os.path.expandvars(value) if isinstance(value, str) else value).expanduser()
    if base_dir is not None:
        path = base_dir / path
    return path.resolve()
