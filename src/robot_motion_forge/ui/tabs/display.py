"""显示页：碰撞几何、重心、支撑区域与明暗主题切换。"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple

from viser import Icon

if TYPE_CHECKING:
    from ..controls import Controls


class DisplayState(NamedTuple):
    collisions: bool
    center_of_mass: bool
    support: bool


class DisplayTab:
    def __init__(self, controls: Controls) -> None:
        self._controls = controls
        self._collision_visible = controls.gui.add_checkbox("显示碰撞几何", initial_value=False)
        self._com_visible = controls.gui.add_checkbox("显示重心", initial_value=True)
        self._support_visible = controls.gui.add_checkbox("显示支撑区域", initial_value=True)
        self._theme_button = controls.gui.add_button("暗色主题")

    def bind_actions(self, *, change_theme: Callable[[], None]) -> None:
        self._controls.click(self._theme_button, change_theme)

    @property
    def state(self) -> DisplayState:
        return DisplayState(self._collision_visible.value, self._com_visible.value, self._support_visible.value)

    def set_theme(self, *, dark_mode: bool) -> None:
        self._theme_button.label = "暗色主题" if dark_mode else "浅色主题"
        self._theme_button.icon = Icon.MOON if dark_mode else Icon.SUN
        self._theme_button.color = None if dark_mode else "gray"
