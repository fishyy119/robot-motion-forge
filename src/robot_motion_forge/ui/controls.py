"""共用控件与事件入队；程序赋值不触发编辑操作。"""

from __future__ import annotations

from collections.abc import Callable, Coroutine, Mapping
from functools import partial
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from viser import (
        ClientHandle,
        GuiApi,
        GuiButtonGroupHandle,
        GuiButtonHandle,
        GuiInputHandle,
        GuiSliderHandle,
    )

    from robot_motion_forge.robot.definition import ControlMode
    from robot_motion_forge.robot.model import Robot

CONTROL_MODES: dict[str, ControlMode] = {"位置": "position", "朝向": "orientation", "位姿": "pose"}


def grouped_joint_names(robot: Robot) -> dict[str, list[str]]:
    groups = {group: names.copy() for group, names in robot.config.joint_groups.items()}
    assigned = {name for names in groups.values() for name in names}
    remaining = [name for name in robot.joint_names if name not in assigned]
    if remaining:
        groups.setdefault("其他关节", []).extend(remaining)
    return {group: names for group, names in groups.items() if names}


class _ClientEvent[HandleT](Protocol):
    @property
    def client(self) -> ClientHandle | None: ...

    @property
    def target(self) -> HandleT: ...


class Controls:
    def __init__(self, gui: GuiApi, enqueue: Callable[[Callable[[], None], ClientHandle], None]) -> None:
        self.gui = gui
        self.enqueue = enqueue

    def client_action[HandleT](
        self, prepare: Callable[[ClientHandle, HandleT], Callable[[], None] | None]
    ) -> Callable[[_ClientEvent[HandleT]], Coroutine[None, None, None]]:
        async def callback(event: _ClientEvent[HandleT]) -> None:
            # 在 Viser 接收线程中取值并入队，避免线程池里的延迟回调读取下一条消息的控件值。
            if event.client is not None:
                action = prepare(event.client, event.target)
                if action is not None:
                    self.enqueue(action, event.client)

        return callback

    def click(self, handle: GuiButtonHandle, action: Callable[[], None]) -> None:
        def prepare(_client: ClientHandle, _target: GuiButtonHandle) -> Callable[[], None]:
            return action

        handle.on_click(self.client_action(prepare))

    def click_group(self, handle: GuiButtonGroupHandle, actions: Mapping[str, Callable[[], None]]) -> None:
        def prepare(_client: ClientHandle, target: GuiButtonGroupHandle) -> Callable[[], None]:
            # 按钮组在入队时确定操作，避免后续点击改变组值后执行了另一项操作。
            return actions[target.value]

        handle.on_click(self.client_action(prepare))

    def input[ValueT](self, handle: GuiInputHandle[ValueT], action: Callable[[ValueT], None]) -> None:
        def prepare(_client: ClientHandle, target: GuiInputHandle[ValueT]) -> Callable[[], None]:
            return partial(action, target.value)

        handle.on_update(self.client_action(prepare))

    @staticmethod
    def sync_slider[ValueT: float | int](handle: GuiSliderHandle[ValueT], value: ValueT) -> None:
        """按真实值扩展滑块范围，再同步数值，避免显示端截断。"""
        handle.min, handle.max = min(handle.min, value), max(handle.max, value)
        handle.value = value

    def button(
        self,
        label: str,
        *,
        primary: bool = False,
        disabled: bool = False,
        visible: bool = True,
    ) -> GuiButtonHandle:
        return self.gui.add_button(label, color=None if primary else "gray", disabled=disabled, visible=visible)
