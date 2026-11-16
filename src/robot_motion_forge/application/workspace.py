"""机器人仓库内的工程目录与文件名解析。"""

from pathlib import Path


class Workspace:
    def __init__(self, repository_root: Path) -> None:
        self.project_dir = repository_root.expanduser().resolve() / ".robot-motion-forge"

    def project_path(self, name: str | None = None) -> Path:
        """生成固定工程目录内的 JSON 路径，文件名省略扩展名时补齐。"""
        name = "untitled.motion.json" if name is None else name.strip()
        if not name or name in {".", ".."}:
            raise ValueError("工程文件名不能为空或为 .、..")
        if any(separator in name for separator in ("/", "\\", "\0")):
            raise ValueError("工程文件名不能包含目录分隔符或空字符")
        if not name.lower().endswith(".json"):
            name += ".json"
        directory = self.project_dir.resolve()
        path = (directory / name).resolve()
        if path.parent != directory:
            raise ValueError("工程文件必须位于机器人仓库的 .robot-motion-forge 目录")
        return path
