"""时间轴页：关键帧刻度、进度拖拽与播放。"""

from __future__ import annotations

import json
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

from ..viser_bridge import gui_uuid

if TYPE_CHECKING:
    from viser import ClientHandle, GuiButtonGroupHandle, GuiSliderHandle, GuiTabHandle

    from robot_motion_forge.motion.draft import PoseSource
    from robot_motion_forge.motion.project import Project

    from ..controls import Controls


_POSE_SOURCES: dict[str, PoseSource] = {"前一关键帧": "previous_keyframe", "轨迹": "trajectory"}


def _marks(project: Project) -> tuple[tuple[float, str, str], ...]:
    return tuple(
        (key.time, key.name or "关键帧", key.id) for key in sorted(project.keyframes, key=lambda key: key.time)
    )


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
        marks = _marks(project)
        self._layout = (project.duration, marks)
        self._scrub = self._add_scrub(project.duration, marks, cursor)
        with tab:
            self._mark_click = controls.gui.add_button_group(
                "定位关键帧", options=tuple(key_id for _, _, key_id in marks), visible=False
            )
            self._mark_metadata = controls.gui.add_html(self._mark_html())
            self._source = controls.gui.add_dropdown("姿态来源", options=tuple(_POSE_SOURCES))
            self._play_button = controls.gui.add_button("播放 / 暂停")

    def bind_actions(
        self,
        *,
        seek: Callable[[float], None],
        select_key: Callable[[str], None],
        set_source: Callable[[PoseSource], None],
        toggle_play: Callable[[], None],
    ) -> None:
        self._seek = seek
        self._controls.input(self._scrub, seek)
        self._controls.input(self._source, lambda label: set_source(_POSE_SOURCES[label]))
        self._controls.click(self._play_button, toggle_play)

        def select_mark(key_id: str) -> None:
            if key_id in self._mark_click.options:
                select_key(key_id)

        def prepare_mark(_client: ClientHandle, target: GuiButtonGroupHandle) -> Callable[[], None]:
            return partial(select_mark, target.value)

        self._mark_click.on_click(self._controls.client_action(prepare_mark))

    def _mark_html(self) -> str:
        # 刻度标签按关键帧 ID 提交，重复点击同一时刻仍执行选帧。
        slider = gui_uuid(self._scrub)
        button_group = gui_uuid(self._mark_click)
        key_ids = json.dumps(self._mark_click.options).replace("<", "\\u003c")
        return (
            f'<script type="application/json" data-rmf-timeline="{slider}" data-mark-click="{button_group}">'
            f"{key_ids}</script>"
        )

    def sync_cursor(self, cursor: float, source: PoseSource) -> None:
        self._scrub.value = cursor
        self._source.value = next(label for label, value in _POSE_SOURCES.items() if value == source)

    def _add_scrub(
        self,
        duration: float,
        marks: tuple[tuple[float, str, str], ...],
        cursor: float,
        *,
        order: float | None = None,
    ) -> GuiSliderHandle[float]:
        with self._tab:
            handle = self._controls.gui.add_slider(
                "时间 / s",
                min=0.0,
                max=duration,
                step=0.05,
                initial_value=min(cursor, duration),
                marks=tuple((time, name) for time, name, _ in marks),
                order=order,
            )
            handle.precision = max(3, handle.precision)
            if self._seek is not None:
                self._controls.input(handle, self._seek)
            return handle

    def refresh(self, project: Project, cursor: float) -> None:
        marks = _marks(project)
        if self._layout == (project.duration, marks):
            return
        self._layout = (project.duration, marks)
        order = self._scrub.order
        self._scrub.remove()
        self._scrub = self._add_scrub(project.duration, marks, cursor, order=order)
        self._mark_click.options = tuple(key_id for _, _, key_id in marks)
        self._mark_metadata.content = self._mark_html()
