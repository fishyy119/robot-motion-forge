"""关键帧编辑、时间定位与播放。"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robot_motion_forge.editor import Editor
    from robot_motion_forge.motion.draft import PoseSource

    from ..notifications import Notifications


class TimelineController:
    def __init__(self, editor: Editor, *, reset_target: Callable[[], None], notifications: Notifications) -> None:
        self._editor, self._reset_target = editor, reset_target
        self._notifications = notifications
        self._playing = False

    def pause(self) -> None:
        self._playing = False

    def set_duration(self, duration: float) -> None:
        self._reset_target()
        self._editor.set_timing(duration=duration)

    def _begin_key_edit(self) -> None:
        self.pause()
        self._reset_target()

    def select_key(self, key_id: str | None) -> None:
        self._begin_key_edit()
        self._editor.select_key(key_id)

    def capture(self) -> None:
        self._begin_key_edit()
        self._editor.capture("关键帧")
        self._notifications.show("关键帧已添加")

    def update_key(
        self,
        key_id: str,
        *,
        time: float | None = None,
        name: str | None = None,
        locked: bool | None = None,
        zero_velocity: bool | None = None,
    ) -> None:
        self._begin_key_edit()
        self._editor.update_key(key_id, time=time, name=name, locked=locked, zero_velocity=zero_velocity)

    def delete_key(self, key_id: str) -> None:
        self._begin_key_edit()
        self._editor.delete_key(key_id)

    def history(self, redo: bool) -> None:
        self._begin_key_edit()
        (self._editor.redo if redo else self._editor.undo)()

    def toggle(self) -> None:
        self._playing = not self._playing
        if self._playing:
            self._editor.select_key(None)
            if self._editor.draft_snapshot().cursor >= self._editor.duration:
                self._editor.seek(0.0)
        self._reset_target()

    def seek(self, time: float) -> None:
        self.pause()
        self._editor.scrub(time)
        self._reset_target()

    def set_source(self, source: PoseSource) -> None:
        self.pause()
        self._editor.set_pose_source(source)
        self._reset_target()

    def tick(self, elapsed: float) -> None:
        if self._playing:
            self._playing = self._editor.advance(elapsed)
            self._reset_target()
