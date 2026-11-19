"""播放状态、时间定位与逐帧推进。"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robot_motion_forge.application.editor import Editor
    from robot_motion_forge.motion.draft import PoseSource


class PlaybackController:
    def __init__(self, editor: Editor, *, reset_target: Callable[[], None]) -> None:
        self._editor, self._reset_target = editor, reset_target
        self._playing = False

    def pause(self) -> None:
        self._playing = False

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
