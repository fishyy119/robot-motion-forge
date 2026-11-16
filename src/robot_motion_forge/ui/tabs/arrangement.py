"""编排页：关键帧编辑、历史操作与时间编排概览。"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from html import escape
from typing import TYPE_CHECKING, Protocol

from robot_motion_forge.motion.project import Project

if TYPE_CHECKING:
    from viser import ClientHandle, GuiDropdownHandle, GuiInputHandle

    from robot_motion_forge.motion.types import Keyframe

    from ..controls import Controls


class ArrangementActions(Protocol):
    def capture(self) -> None: ...
    def delete_key(self) -> None: ...
    def history(self, redo: bool) -> None: ...
    def select_key(self, key_id: str | None) -> None: ...
    def set_name(self, key_id: str, value: str) -> None: ...
    def set_time(self, key_id: str, value: float) -> None: ...
    def set_locked(self, key_id: str, value: bool) -> None: ...


class ArrangementTab:
    def __init__(self, controls: Controls) -> None:
        gui = controls.gui
        self._controls = controls
        self._key_labels: dict[str, str] = {}
        empty_options: list[str] = ["—"]
        with gui.add_folder("关键帧"):
            self._key_select: GuiDropdownHandle[str] = gui.add_dropdown("选中帧", options=empty_options)
            self._key_name = gui.add_text("名称", initial_value="", disabled=True)
            self._key_time = gui.add_number("帧时间 / s", initial_value=0.0, step=0.01, disabled=True)
            self._key_locked = gui.add_checkbox("锁定姿态", initial_value=True, disabled=True)
            self._add_key_button = controls.button("添加关键帧", primary=True)
            self._delete_key_button = controls.button("删除关键帧", disabled=True)
            self._history = gui.add_button_group("历史", options=("撤销", "重做"))
        with gui.add_folder("时间编排"):
            self._arrangement = gui.add_html("")

    def bind_actions(self, actions: ArrangementActions) -> None:
        controls = self._controls
        controls.click(self._add_key_button, actions.capture)
        controls.click(self._delete_key_button, actions.delete_key)
        controls.click_group(
            self._history, {"撤销": lambda: actions.history(False), "重做": lambda: actions.history(True)}
        )

        def prepare_selection(_client: ClientHandle, _target: GuiDropdownHandle[str]) -> Callable[[], None]:
            return partial(actions.select_key, self._key_id)

        self._key_select.on_update(controls.client_action(prepare_selection))
        self._bind_property(controls, self._key_name, actions.set_name)
        self._bind_property(controls, self._key_time, actions.set_time)
        self._bind_property(controls, self._key_locked, actions.set_locked)

    @property
    def _key_id(self) -> str | None:
        return self._key_labels.get(self._key_select.value)

    def _bind_property[ValueT: str | float | bool](
        self, controls: Controls, handle: GuiInputHandle[ValueT], update: Callable[[str, ValueT], None]
    ) -> None:
        def prepare(_client: ClientHandle, target: GuiInputHandle[ValueT]) -> Callable[[], None] | None:
            if (key_id := self._key_id) is not None:
                # 入队时保存所属关键帧和字段值，避免切帧或界面刷新覆盖尚未处理的输入。
                return partial(update, key_id, target.value)
            return None

        handle.on_update(controls.client_action(prepare))

    def refresh(self, project: Project, *, selected_key: str | None, cursor: float) -> None:
        keys = sorted(project.keyframes, key=lambda k: k.time)
        self._key_labels = {f"{k.time:.3f}s · {k.name}": k.id for k in keys}
        if len(self._key_labels) != len(keys):
            # 毫秒显示造成重名时保留完整时间精度，避免选择项相互覆盖。
            self._key_labels = {f"{k.time}s · {k.name}": k.id for k in keys}
        options = ("—", *self._key_labels)
        if self._key_select.options != options:
            self._key_select.options = options
        self.sync_key(project, selected_key=selected_key, cursor=cursor)
        content = _render_arrangement(project)
        if self._arrangement.content != content:
            self._arrangement.content = content

    def sync_key(self, project: Project, *, selected_key: str | None, cursor: float) -> None:
        self._key_select.value = next(
            (label for label, ident in self._key_labels.items() if ident == selected_key), "—"
        )
        key: Keyframe | None = next((key for key in project.keyframes if key.id == selected_key), None)
        self._key_name.disabled = self._key_time.disabled = self._key_locked.disabled = key is None
        self._delete_key_button.disabled = key is None
        self._add_key_button.disabled = any(abs(key.time - cursor) < 1e-8 for key in project.keyframes)
        if key is None:
            self._key_name.value, self._key_time.value, self._key_locked.value = "", cursor, True
        else:
            self._key_name.value, self._key_time.value, self._key_locked.value = key.name, key.time, key.locked


_STYLE = """
<style>
[data-rmf-arrangement] {
    padding: 0.5rem 0.75rem;
    font-size: var(--rmf-font-control);
    line-height: 1.5;
    font-variant-numeric: tabular-nums;
}
[data-rmf-arrangement] table {
    width: 100%;
    table-layout: fixed;
    border-collapse: collapse;
}
[data-rmf-arrangement] th, [data-rmf-arrangement] td {
    padding: 0.4rem 0.5rem;
    border-bottom: 1px solid var(--mantine-color-default-border);
    text-align: left;
    overflow-wrap: anywhere;
}
[data-rmf-arrangement] th {
    color: var(--mantine-color-dimmed);
    font-weight: 500;
}
[data-rmf-arrangement] .rmf-key-time {
    width: 6rem;
    text-align: right;
    white-space: nowrap;
}
[data-rmf-arrangement] .rmf-key-lock {
    width: 3rem;
    text-align: center;
}
</style>
"""


def _render_arrangement(project: Project) -> str:
    """时间编排概览：对齐显示关键帧时间、名称与锁定状态。"""
    parts = [
        _STYLE,
        "<section data-rmf-arrangement>",
    ]
    parts.extend(
        [
            '<table aria-label="关键帧编排">',
            '<thead><tr><th class="rmf-key-time" scope="col">时间 / s</th>',
            '<th scope="col">关键帧</th><th class="rmf-key-lock" scope="col">锁定</th></tr></thead>',
            "<tbody>",
        ]
    )
    for key in sorted(project.keyframes, key=lambda key: key.time):
        state = "锁定" if key.locked else "未锁定"
        parts.append(
            f'<tr><td class="rmf-key-time">{key.time:.3f}</td>'
            f"<td>{escape(key.name)}</td>"
            f'<td class="rmf-key-lock" aria-label="{state}">{"●" if key.locked else "○"}</td></tr>'
        )
    parts.append("</tbody></table>")
    parts.append("</section>")
    return "".join(parts)
