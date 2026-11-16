"""工程保存、重新加载与浏览器数据导出。"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from robot_motion_forge.application.editor import Editor
from robot_motion_forge.application.workspace import Workspace
from robot_motion_forge.io.export import serialize_motion
from robot_motion_forge.io.project import load_project

if TYPE_CHECKING:
    from pathlib import Path

    from viser import ClientHandle

    from ..notifications import Notifications
    from ..tabs.project import ProjectTab


class ProjectController:
    def __init__(
        self,
        editor: Editor,
        tab: ProjectTab,
        *,
        workspace: Workspace,
        replace_project: Callable[[Editor, Path], None],
        notifications: Notifications,
    ) -> None:
        self._editor, self._tab = editor, tab
        self._workspace = workspace
        self._replace_project, self._notifications = replace_project, notifications

    def save(self, filename: str) -> None:
        path = self._workspace.project_path(filename)
        self._editor.save(path)
        # 只规范化仍对应本次保存的文件名，保留排队期间的新输入。
        if self._tab.filename == filename:
            self._tab.filename = path.name
        self._notifications.show("工程已保存", path.name)

    def load(self, filename: str) -> None:
        path = self._workspace.project_path(filename)
        loaded = load_project(path, self._editor.robot)
        editor = Editor(self._editor.robot, loaded.project, loaded.trajectory, draft=loaded.draft)
        self._replace_project(editor, path)
        self._notifications.show("工程已重新加载", path.name)

    def export(self, client: ClientHandle) -> None:
        state = self._editor.snapshot()
        payload = serialize_motion(self._editor.robot, state.project, self._editor.preview())
        # Viser 下载流程自带接收完成通知。
        client.send_file_download("motion.pkl", payload, save_immediately=True)
