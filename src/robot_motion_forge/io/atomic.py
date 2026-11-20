"""原子文件写入，供工程保存与机器人定义 Schema 导出共用。"""

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as stream:
            temporary = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    """原子写入 JSON，供工程保存与机器人定义 Schema 导出共用。"""
    content = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    atomic_write(path, content)
