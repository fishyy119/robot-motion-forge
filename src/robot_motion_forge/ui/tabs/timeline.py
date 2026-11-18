"""时间轴页：关键帧刻度、进度拖拽与播放。"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from viser import GuiSliderHandle, GuiTabHandle

    from robot_motion_forge.application.editor import PoseSource
    from robot_motion_forge.motion.project import Project

    from ..controls import Controls


_POSE_SOURCES: dict[str, PoseSource] = {"前一关键帧": "previous_keyframe", "轨迹": "trajectory"}


class TimelineTab:
    def __init__(
        self,
        controls: Controls,
        *,
        tab: GuiTabHandle,
        project: Project,
        cursor: float,
    ) -> None:
        self._controls, self._tab = controls, tab
        self._seek: Callable[[float], None] | None = None
        self._scrub = self._add_scrub(project, cursor)
        with tab:
            self._source = controls.gui.add_dropdown("姿态来源", options=tuple(_POSE_SOURCES))
            self._play_button = controls.gui.add_button("播放 / 暂停")

    def bind_actions(
        self,
        *,
        seek: Callable[[float], None],
        set_source: Callable[[PoseSource], None],
        toggle_play: Callable[[], None],
    ) -> None:
        self._seek = seek
        self._controls.input(self._scrub, seek)
        self._controls.input(self._source, lambda label: set_source(_POSE_SOURCES[label]))
        self._controls.click(self._play_button, toggle_play)

    @property
    def snap_tolerance(self) -> float:
        # 拖拽与手输共用回调，按毫秒输入精度吸附到关键帧。
        return min(0.001, self._scrub.step) / 2

    def sync_cursor(self, cursor: float) -> None:
        self._scrub.value = cursor

    def _add_scrub(self, project: Project, cursor: float, *, order: float | None = None) -> GuiSliderHandle[float]:
        marks = tuple((key.time, key.name) for key in sorted(project.keyframes, key=lambda k: k.time))
        self._layout = (project.duration, marks)
        with self._tab:
            handle = self._controls.gui.add_slider(
                "时间 / s",
                min=0.0,
                max=project.duration,
                step=0.05,
                initial_value=min(cursor, project.duration),
                marks=marks,
                order=order,
            )
            handle.precision = max(3, handle.precision)
            if self._seek is not None:
                self._controls.input(handle, self._seek)
            return handle

    def refresh(self, project: Project, cursor: float) -> None:
        marks = tuple((key.time, key.name) for key in sorted(project.keyframes, key=lambda k: k.time))
        if self._layout == (project.duration, marks):
            return
        order = self._scrub.order
        self._scrub.remove()
        self._scrub = self._add_scrub(project, cursor, order=order)
