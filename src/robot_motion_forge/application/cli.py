"""网页编辑器的命令行启动入口。"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from robot_motion_forge.io.project import load_project
from robot_motion_forge.robot.model import Robot

from .editor import Editor
from .workspace import Workspace

DEFAULT_ROBOT_CONFIG = Path("configs/robots/robot.json")


class CLIArgs(argparse.Namespace):
    robot: Path
    project: str | None
    host: str
    port: int


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="robot-motion-forge",
        description="启动 Robot Motion Forge 网页编辑器",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--robot", type=Path, default=DEFAULT_ROBOT_CONFIG, help="机器人定义文件；运行期间固定")
    parser.add_argument(
        "--project", metavar="NAME", help="启动时加载的工程文件名；目录固定为机器人仓库的 .robot-motion-forge"
    )
    parser.add_argument("--host", default="127.0.0.1", help="服务监听地址")
    parser.add_argument("--port", type=int, default=8080, help="服务监听端口")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv, namespace=CLIArgs())
    try:
        robot = Robot.load(args.robot)
        workspace = Workspace(robot.repository_root)
        project_path = workspace.project_path(args.project) if args.project is not None else None
        loaded = load_project(project_path, robot) if project_path is not None else None
        try:
            from robot_motion_forge.ui.app import ForgeApp

            if loaded is None:
                editor = Editor(robot)
            else:
                editor = Editor(robot, loaded.project, loaded.trajectory, draft=loaded.draft)
            app = ForgeApp(
                editor,
                workspace=workspace,
                host=args.host,
                port=args.port,
                project_name=args.project,
            )
        except ModuleNotFoundError as exc:
            raise ValueError(f"缺少界面依赖 {exc.name}；请在 mujoco-py12 中运行 python -m pip install -e .") from exc
        app.run()
    except (ValueError, OSError, KeyError, TypeError, StopIteration) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
