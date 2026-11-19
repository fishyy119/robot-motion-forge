"""编排页：关键帧编辑、历史操作与时间编排概览。"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from html import escape
from typing import TYPE_CHECKING, NamedTuple, Protocol

from robot_motion_forge.motion.project import Project

if TYPE_CHECKING:
    from viser import ClientHandle, GuiCheckboxHandle, GuiDropdownHandle, GuiNumberHandle, GuiTextHandle

    from robot_motion_forge.motion.types import Keyframe

    from ..controls import Controls


class ArrangementActions(Protocol):
    def capture(self) -> None: ...
    def delete_key(self, key_id: str) -> None: ...
    def history(self, redo: bool) -> None: ...
    def select_key(self, key_id: str | None) -> None: ...
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
    locked: GuiCheckboxHandle
    zero_velocity: GuiCheckboxHandle


class ArrangementTab:
    def __init__(self, controls: Controls) -> None:
        gui = controls.gui
        self._controls = controls
        self._key_labels: dict[str, str] = {}
        self._actions: ArrangementActions | None = None
        empty_options: list[str] = ["—"]
        with gui.add_folder("关键帧") as self._key_folder:
            self._key_select: GuiDropdownHandle[str] = gui.add_dropdown("选中帧", options=empty_options)
            self._fields = self._add_fields(None, 0.0)
            self._add_key_button = controls.button("添加关键帧", primary=True)
            self._delete_key_button = controls.button("删除关键帧", disabled=True)
            self._history = gui.add_button_group("历史", options=("撤销", "重做"))
        with gui.add_folder("时间编排"):
            self._arrangement = gui.add_html("")

    def bind_actions(self, actions: ArrangementActions) -> None:
        self._actions = actions
        controls = self._controls
        controls.click(self._add_key_button, actions.capture)
        controls.click_group(
            self._history, {"撤销": lambda: actions.history(False), "重做": lambda: actions.history(True)}
        )

        def prepare_selection(_client: ClientHandle, target: GuiDropdownHandle[str]) -> Callable[[], None]:
            return partial(actions.select_key, self._key_labels.get(target.value))

        self._key_select.on_update(controls.client_action(prepare_selection))
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
            gui.add_checkbox(
                "锁定姿态",
                initial_value=key.locked if key else True,
                disabled=key is None,
                order=old.locked.order if old else None,
            ),
            gui.add_checkbox(
                "速度归零",
                initial_value=key.zero_velocity if key else False,
                disabled=key is None,
                order=old.zero_velocity.order if old else None,
            ),
        )

    def _bind_fields(self) -> None:
        fields, actions = self._fields, self._actions
        key_id = fields.key_id
        if key_id is None or actions is None:
            return
        self._controls.input(fields.name, lambda value: actions.update_key(key_id, name=value))
        self._controls.input(fields.time, lambda value: actions.update_key(key_id, time=value))
        self._controls.input(fields.locked, lambda value: actions.update_key(key_id, locked=value))
        self._controls.input(fields.zero_velocity, lambda value: actions.update_key(key_id, zero_velocity=value))
        self._controls.click(self._delete_key_button, lambda: actions.delete_key(key_id))

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
        self._add_key_button.disabled = any(abs(key.time - cursor) < 1e-8 for key in project.keyframes)
        if self._fields.key_id != (key.id if key else None):
            # 控件身份与关键帧绑定；切帧后，已入队的编辑仍落在发起时的关键帧。
            old = self._fields
            order = self._delete_key_button.order
            old.name.remove()
            old.time.remove()
            old.locked.remove()
            old.zero_velocity.remove()
            self._delete_key_button.remove()
            with self._key_folder:
                self._fields = self._add_fields(key, cursor, old)
                self._delete_key_button = self._controls.gui.add_button(
                    "删除关键帧", color="gray", disabled=key is None, order=order
                )
            self._bind_fields()
        if key is None:
            self._fields.time.value = cursor
        else:
            self._fields.name.value, self._fields.time.value, self._fields.locked.value = key.name, key.time, key.locked
            self._fields.zero_velocity.value = key.zero_velocity


def _render_arrangement(project: Project) -> str:
    """时间编排概览：对齐显示关键帧时间、名称与约束状态。"""
    parts = [
        "<section data-rmf-arrangement>",
        '<table aria-label="关键帧编排">',
        '<thead><tr><th class="rmf-key-time" scope="col">时间 / s</th>',
        '<th scope="col">关键帧</th><th class="rmf-key-lock" scope="col">锁定</th>',
        '<th class="rmf-key-lock" scope="col" title="速度归零">归零</th></tr></thead>',
        "<tbody>",
    ]
    for key in sorted(project.keyframes, key=lambda key: key.time):
        state = "锁定" if key.locked else "未锁定"
        velocity_state = "速度归零" if key.zero_velocity else "未指定速度归零"
        parts.append(
            f'<tr><td class="rmf-key-time">{key.time:.3f}</td>'
            f"<td>{escape(key.name)}</td>"
            f'<td class="rmf-key-lock" aria-label="{state}">{"●" if key.locked else "○"}</td>'
            f'<td class="rmf-key-lock" aria-label="{velocity_state}">{"●" if key.zero_velocity else "○"}</td></tr>'
        )
    parts.append("</tbody></table>")
    parts.append("</section>")
    return "".join(parts)
