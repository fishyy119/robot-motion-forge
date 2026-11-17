"""工程页：存储位置、文件名、保存与重新加载。"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from html import escape
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pathlib import Path

    from viser import ClientHandle, GuiButtonGroupHandle

    from ..controls import Controls


class ProjectActions(Protocol):
    def save(self, filename: str) -> None: ...
    def load(self, filename: str) -> None: ...


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

    def bind_actions(self, actions: ProjectActions) -> None:
        controls = self._controls
        file_actions = {"保存": actions.save, "重新加载": actions.load}

        def prepare_file(_client: ClientHandle, target: GuiButtonGroupHandle) -> Callable[[], None]:
            return partial(file_actions[target.value], self.filename)

        self._file_actions.on_click(controls.client_action(prepare_file))

    @property
    def filename(self) -> str:
        return self._filename.value

    @filename.setter
    def filename(self, value: str) -> None:
        self._filename.value = value
