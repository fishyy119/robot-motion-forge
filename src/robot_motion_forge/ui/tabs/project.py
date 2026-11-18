"""工程页：存储位置、文件名、保存与重新加载。"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from html import escape
from typing import TYPE_CHECKING

from robot_motion_forge.application.editor import Editor

if TYPE_CHECKING:
    from pathlib import Path

    from viser import ClientHandle, GuiButtonGroupHandle

    from robot_motion_forge.application.workspace import Workspace

    from ..controls import Controls
    from ..notifications import Notifications


class ProjectTab:
    def __init__(self, controls: Controls, *, project_path: Path) -> None:
        gui = controls.gui
        self._controls = controls
        with gui.add_folder("工程文件"):
            gui.add_html(
                f'<dl class="rmf-project-directory"><dt>工程目录</dt><dd>{escape(str(project_path.parent))}</dd></dl>'
            )
            self._filename = gui.add_text("文件名", initial_value=project_path.name)
            self._file_actions = gui.add_button_group("工程操作", options=("保存", "重新加载"))

    def bind_actions(
        self,
        editor: Editor,
        *,
        workspace: Workspace,
        replace_project: Callable[[Editor, Path], None],
        notifications: Notifications,
    ) -> None:
        def save(filename: str) -> None:
            path = workspace.project_path(filename)
            editor.save(path)
            # 只规范化仍对应本次保存的文件名，保留排队期间的新输入。
            if self._filename.value == filename:
                self._filename.value = path.name
            notifications.show("工程已保存", path.name)

        def load(filename: str) -> None:
            path = workspace.project_path(filename)
            replace_project(Editor.load(editor.robot, path), path)
            notifications.show("工程已重新加载", path.name)

        file_actions = {"保存": save, "重新加载": load}

        def prepare_file(_client: ClientHandle, target: GuiButtonGroupHandle) -> Callable[[], None]:
            return partial(file_actions[target.value], self._filename.value)

        self._file_actions.on_click(self._controls.client_action(prepare_file))
