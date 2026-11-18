"""关键帧与编辑历史的交互流程。"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robot_motion_forge.application.editor import Editor

    from ..notifications import Notifications


class ArrangementController:
    def __init__(
        self,
        editor: Editor,
        *,
        pause: Callable[[], None],
        reset_target: Callable[[], None],
        notifications: Notifications,
    ) -> None:
        self._editor = editor
        self._pause = pause
        self._reset_target = reset_target
        self._notifications = notifications

    def select_key(self, key_id: str | None) -> None:
        self._pause()
        self._reset_target()
        self._editor.select_key(key_id)

    def capture(self) -> None:
        self._pause()
        self._reset_target()
        self._editor.capture("关键帧")
        self._notifications.show("关键帧已添加")

    def update_key(
        self, key_id: str, *, time: float | None = None, name: str | None = None, locked: bool | None = None
    ) -> None:
        self._pause()
        self._reset_target()
        self._editor.update_key(key_id, time=time, name=name, locked=locked)

    def delete_key(self) -> None:
        if (key_id := self._editor.draft_snapshot().selected_key) is None:
            raise ValueError("请先选择关键帧")
        self._pause()
        self._reset_target()
        self._editor.delete_key(key_id)

    def history(self, redo: bool) -> None:
        self._pause()
        self._reset_target()
        (self._editor.redo if redo else self._editor.undo)()
