"""Viser 原生主题与机器人配色。"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from viser import GuiApi


def accent_color(dark_mode: bool) -> tuple[int, int, int]:
    return (45, 212, 191) if dark_mode else (15, 118, 110)


def warning_color(dark_mode: bool) -> tuple[int, int, int]:
    return (255, 183, 77) if dark_mode else (180, 83, 9)


def robot_color(dark_mode: bool) -> tuple[int, int, int]:
    return (138, 145, 155) if dark_mode else (160, 164, 171)


def locked_color(dark_mode: bool) -> tuple[int, int, int]:
    return (96, 165, 250) if dark_mode else (30, 58, 138)


def apply_theme(gui: GuiApi, *, dark_mode: bool) -> None:
    gui.configure_theme(
        dark_mode=dark_mode,
        control_width="large",
        show_logo=False,
        show_share_button=False,
        brand_color=accent_color(dark_mode),
    )
