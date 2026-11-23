"""网页编辑器的命令行启动入口。"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

DEFAULT_ROBOT_CONFIG = "configs/robots/robot.json"


class CLIArgs(argparse.Namespace):
    robot: str
    project: str | None
    host: str
    port: int


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="robot-motion-forge",
        description="启动 Robot Motion Forge 网页编辑器",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--robot", default=DEFAULT_ROBOT_CONFIG, help="机器人定义文件路径，支持 ~ 和环境变量")
    parser.add_argument(
        "--project",
        metavar="PATH",
        help="启动时加载的工程路径。@/ 相对于 robot.project_root，其余相对路径以当前目录为基准。目标必须位于工程根目录内",
    )
    parser.add_argument("--host", default="127.0.0.1", help="服务监听地址")
    parser.add_argument("--port", type=int, default=8080, help="服务监听端口")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    # 帮助与参数校验在加载应用依赖前完成。
    args = _parser().parse_args(argv, namespace=CLIArgs())

    from .editor import Editor
    from .io.project_file import resolve_project_path
    from .robot.model import Robot

    try:
        robot = Robot.load(args.robot)
        project_input = None
        if (project_arg := args.project) is not None:
            project_input = (
                Path(project_arg.removeprefix("@/").lstrip("/"))
                if project_arg.startswith("@/")
                else Path(project_arg).absolute()
            )
        project_path = resolve_project_path(robot.project_root, project_input)
        editor = Editor.load(robot, project_path) if project_input is not None else Editor(robot)
    except (ValueError, OSError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1

    from .ui.app import ForgeApp

    app = ForgeApp(
        editor,
        project_path=project_path,
        host=args.host,
        port=args.port,
    )
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
