"""Viser 浏览器扩展的注册，以及私有控件标识的访问。"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from viser import _messages

if TYPE_CHECKING:
    from viser import GuiButtonGroupHandle, GuiInputHandle, ViserServer


def gui_uuid[ValueT](handle: GuiInputHandle[ValueT] | GuiButtonGroupHandle) -> str:
    return handle._impl.uuid  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]


def register_browser_extensions(server: ViserServer) -> None:
    root = Path(__file__).parent
    scripts = ("viser_bridge.js", "scene.js", "tabs/timeline.js", "tabs/joint_curves.js")
    code = "\n".join((root / name).read_text(encoding="utf-8") for name in scripts)
    source = f"(() => {{\n{code}\n}})();"
    server._websock_server.queue_message(_messages.RunJavascriptMessage(source=source))  # pyright: ignore[reportPrivateUsage]
