"""工程页：存储位置、文件名、保存与重新加载。"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from html import escape
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from viser import ClientHandle, GuiButtonGroupHandle

    from ..controls import Controls


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
        *,
        save: Callable[[str], str],
        reload: Callable[[str], None],
    ) -> None:
        def save_file(filename: str) -> None:
            normalized_name = save(filename)
            # 只规范化仍对应本次保存的文件名，保留排队期间的新输入。
            if self._filename.value == filename:
                self._filename.value = normalized_name

        file_actions = {"保存": save_file, "重新加载": reload}

        def prepare_file(_client: ClientHandle, target: GuiButtonGroupHandle) -> Callable[[], None]:
            return partial(file_actions[target.value], self._filename.value)

        self._file_actions.on_click(self._controls.client_action(prepare_file))
