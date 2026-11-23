"""编排页：关键帧编辑、历史操作与时间编排概览。"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import partial
from html import escape
from typing import TYPE_CHECKING, NamedTuple, Protocol
from urllib.parse import quote, unquote

from robot_motion_forge.motion.project import Project

from ..viser_bridge import gui_uuid

if TYPE_CHECKING:
    from viser import ClientHandle, GuiButtonGroupHandle, GuiDropdownHandle, GuiNumberHandle, GuiTextHandle

    from robot_motion_forge.motion.project import Keyframe

    from ..controls import Controls


class ArrangementActions(Protocol):
    def capture(self) -> None: ...
    def delete_key(self, key_id: str) -> None: ...
    def history(self, redo: bool) -> None: ...
    def select_key(self, key_id: str | None) -> None: ...
    def set_duration(self, duration: float) -> None: ...
    def update_key(
        self,
        key_id: str,
        *,
        time: float | None = None,
        name: str | None = None,
        locked: bool | None = None,
        zero_velocity: bool | None = None,
    ) -> None: ...


class _KeyFields(NamedTuple):
    key_id: str | None
    name: GuiTextHandle
    time: GuiNumberHandle[float]


class ArrangementTab:
    def __init__(self, controls: Controls, *, duration: float) -> None:
        gui = controls.gui
        self._controls = controls
        self._key_labels: dict[str, str] = {}
        self._keys: dict[str, Keyframe] = {}
        self._highlighted_key: str | None = None
        self._actions: ArrangementActions | None = None
        empty_options: list[str] = ["—"]
        with gui.add_folder("关键帧") as self._key_folder:
            self._key_select: GuiDropdownHandle[str] = gui.add_dropdown("选中帧", options=empty_options)
            self._fields = self._add_fields(None, 0.0)
            self._add_key_button = controls.button("添加关键帧", primary=True)
            self._delete_key_button = controls.button("删除关键帧", disabled=True)
            self._history = gui.add_button_group("历史", options=("撤销", "重做"))
        with gui.add_folder("时间编排"):
            self._duration = gui.add_number("动作时长 / s", initial_value=duration, step=0.1)
            self._table_click = gui.add_button_group("编排表操作", options=("—",), visible=False)
            self._arrangement = gui.add_html("")

    def bind_actions(self, actions: ArrangementActions) -> None:
        self._actions = actions
        controls = self._controls
        controls.input(self._duration, actions.set_duration)
        controls.click(self._add_key_button, actions.capture)
        controls.click_group(
            self._history, {"撤销": lambda: actions.history(False), "重做": lambda: actions.history(True)}
        )

        def prepare_selection(_client: ClientHandle, target: GuiDropdownHandle[str]) -> Callable[[], None]:
            return partial(actions.select_key, self._key_labels.get(target.value))

        def table_action(value: str) -> None:
            action, _, encoded_id = value.partition(":")
            key_id = unquote(encoded_id)
            key = self._keys.get(key_id)
            if key is None:
                return
            if action == "select":
                actions.select_key(key_id)
            elif action == "locked":
                actions.update_key(key_id, locked=not key.locked)
            elif action == "zero_velocity":
                actions.update_key(key_id, zero_velocity=not key.zero_velocity)

        def prepare_table_action(_client: ClientHandle, target: GuiButtonGroupHandle) -> Callable[[], None]:
            return partial(table_action, target.value)

        self._key_select.on_update(controls.client_action(prepare_selection))
        self._table_click.on_click(controls.client_action(prepare_table_action))
        self._bind_fields()

    def _add_fields(self, key: Keyframe | None, cursor: float, old: _KeyFields | None = None) -> _KeyFields:
        gui = self._controls.gui
        return _KeyFields(
            key.id if key else None,
            gui.add_text(
                "名称",
                initial_value=key.name if key else "",
                disabled=key is None,
                order=old.name.order if old else None,
            ),
            gui.add_number(
                "帧时间 / s",
                initial_value=key.time if key else cursor,
                step=0.01,
                disabled=key is None,
                order=old.time.order if old else None,
            ),
        )

    def _bind_fields(self) -> None:
        fields, actions = self._fields, self._actions
        key_id = fields.key_id
        if key_id is None or actions is None:
            return
        self._controls.input(fields.name, lambda value: actions.update_key(key_id, name=value))
        self._controls.input(fields.time, lambda value: actions.update_key(key_id, time=value))
        self._controls.click(self._delete_key_button, lambda: actions.delete_key(key_id))

    def refresh(self, project: Project, *, selected_key: str | None, cursor: float) -> None:
        self._duration.value = project.duration
        keys = sorted(project.keyframes, key=lambda k: k.time)
        self._keys = {key.id: key for key in keys}
        self._key_labels = {f"{k.time:.3f}s · {k.name}": k.id for k in keys}
        if len(self._key_labels) != len(keys):
            # 毫秒显示造成重名时保留完整时间精度，避免选择项相互覆盖。
            self._key_labels = {f"{k.time}s · {k.name}": k.id for k in keys}
        options = ("—", *self._key_labels)
        if self._key_select.options != options:
            self._key_select.options = options
        table_options = tuple(command for key in keys for command in _table_commands(key.id))
        if self._table_click.options != table_options:
            self._table_click.options = table_options
        self.sync_key(project, selected_key=selected_key, cursor=cursor, refresh_table=True)

    def sync_key(
        self, project: Project, *, selected_key: str | None, cursor: float, refresh_table: bool = False
    ) -> None:
        self._key_select.value = next(
            (label for label, ident in self._key_labels.items() if ident == selected_key), "—"
        )
        key = self._keys.get(selected_key) if selected_key is not None else None
        cursor_key = next((key for key in self._keys.values() if abs(key.time - cursor) <= 1e-8), None)
        current = key or cursor_key
        current_id = current.id if current is not None else None
        if refresh_table or current_id != self._highlighted_key:
            self._highlighted_key = current_id
            content = _render_arrangement(self._keys.values(), gui_uuid(self._table_click), current_id)
            if self._arrangement.content != content:
                self._arrangement.content = content
        time_editable = key is not None and not project.is_boundary_key(key)
        self._add_key_button.disabled = cursor_key is not None
        if self._fields.key_id != (key.id if key else None):
            # 控件身份与关键帧绑定；切帧后，已入队的编辑仍落在发起时的关键帧。
            old = self._fields
            order = self._delete_key_button.order
            old.name.remove()
            old.time.remove()
            self._delete_key_button.remove()
            with self._key_folder:
                self._fields = self._add_fields(key, cursor, old)
                self._delete_key_button = self._controls.gui.add_button(
                    "删除关键帧", color="gray", disabled=not time_editable, order=order
                )
            self._bind_fields()
        self._fields.time.disabled = self._delete_key_button.disabled = not time_editable
        if key is None:
            self._fields.time.value = cursor
        else:
            self._fields.name.value, self._fields.time.value = key.name, key.time


def _table_commands(key_id: str) -> tuple[str, str, str]:
    suffix = quote(key_id, safe="")
    return f"select:{suffix}", f"locked:{suffix}", f"zero_velocity:{suffix}"


def _state_cell(value: str, label: str, enabled: bool) -> str:
    return (
        f'<td class="rmf-key-lock"><button type="button" class="rmf-key-state" data-rmf-key-action="{escape(value)}" '
        f'aria-label="{escape(label)}" aria-pressed="{str(enabled).lower()}">{"●" if enabled else "○"}</button></td>'
    )


def _render_arrangement(keys: Iterable[Keyframe], actions: str, selected_key: str | None) -> str:
    """在同一张表中定位关键帧和编辑其求解状态。"""
    parts = [
        f'<section data-rmf-arrangement="{escape(actions)}">',
        '<table aria-label="关键帧编排">',
        '<thead><tr><th class="rmf-key-time" scope="col">时间 / s</th>',
        '<th scope="col">关键帧</th><th class="rmf-key-lock" scope="col">锁定</th>',
        '<th class="rmf-key-lock" scope="col" title="速度归零">归零</th></tr></thead>',
        "<tbody>",
    ]
    for key in keys:
        name = key.name or "关键帧"
        select, lock, zero = _table_commands(key.id)
        selected = key.id == selected_key
        row_state = " data-rmf-current-key" if selected else ""
        current = ' aria-current="true"' if selected else ""
        parts.extend(
            (
                f'<tr{row_state}><td class="rmf-key-time">{key.time:.3f}</td>',
                f'<td><button type="button" class="rmf-key-name" data-rmf-key-action="{escape(select)}"{current}>'
                f"{escape(name)}</button></td>",
                _state_cell(lock, f"{name} · 锁定姿态", key.locked),
                _state_cell(zero, f"{name} · 速度归零", key.zero_velocity),
                "</tr>",
            )
        )
    parts.append("</tbody></table>")
    parts.append("</section>")
    return "".join(parts)
