"""Viser 浏览器扩展的注册，以及私有控件标识的访问。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from viser import _messages

if TYPE_CHECKING:
    from viser import GuiButtonGroupHandle, GuiInputHandle, ViserServer


def gui_uuid[ValueT](handle: GuiInputHandle[ValueT] | GuiButtonGroupHandle) -> str:
    return handle._impl.uuid  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]


def json_for_script(value: object, *, ensure_ascii: bool = True) -> str:
    """序列化为 application/json 脚本的文本，避免内容被解析为 HTML 标签。"""
    return json.dumps(value, ensure_ascii=ensure_ascii).replace("<", "\\u003c")


def register_browser_extensions(server: ViserServer) -> None:
    root = Path(__file__).parent
    scripts = (
        "viser_bridge.js",
        "layout.js",
        "scene/interactions.js",
        "tabs/arrangement.js",
        "tabs/timeline.js",
        "tabs/joint_curves.js",
    )
    code = "\n".join((root / name).read_text(encoding="utf-8") for name in scripts)
    stylesheet = json_for_script((root / "styles.css").read_text(encoding="utf-8"))
    source = f"((css) => {{\n{code}\n}})({stylesheet});"
    server._websock_server.queue_message(_messages.RunJavascriptMessage(source=source))  # pyright: ignore[reportPrivateUsage]
