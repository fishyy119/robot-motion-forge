"""姿态页：操作对象、拖拽反馈与精确调整。"""

from __future__ import annotations

from collections.abc import Callable, Collection
from functools import partial
from typing import TYPE_CHECKING, Literal, NamedTuple, Protocol

import mujoco as mj
import numpy as np

from robot_motion_forge.robot.model import FloatArray, Robot, rotation

if TYPE_CHECKING:
    from viser import ClientHandle, GuiButtonHandle, GuiDropdownHandle, GuiNumberHandle, GuiSliderHandle

    from robot_motion_forge.solvers.drag import DragResult

    from ..controls import Controls

type TargetMode = Literal["位置", "朝向", "位姿"]


class PoseState(NamedTuple):
    object_name: str | None
    mode: TargetMode | None
    lock_base: bool


class PoseActions(Protocol):
    def reset_target(self) -> None: ...
    def lock_joint(self, name: str | None) -> None: ...
    def unlock_joint(self, name: str | None) -> None: ...
    def unlock_all_joints(self) -> None: ...
    def home(self) -> None: ...
    def set_joint(self, name: str, value: float) -> None: ...
    def set_root_position(self, value: tuple[float, float, float]) -> None: ...
    def set_root_rpy(self, value: tuple[float, float, float]) -> None: ...


class PoseTab:
    def __init__(self, controls: Controls, *, robot: Robot, qpos: FloatArray) -> None:
        gui = controls.gui
        self._controls = controls
        self._robot = robot
        self._drag_feedback = gui.add_markdown("", visible=False)
        self._object_labels = {f"末端 · {name}": name for name in robot.end_effectors}
        self._object_labels.update({f"关节 · {name}": f"joint:{name}" for name in robot.joint_names})
        self._object_select = gui.add_dropdown("操作对象", options=tuple(self._object_labels) or ("—",))
        mode_labels: dict[str, TargetMode] = {"position": "位置", "orientation": "朝向", "pose": "位姿"}
        modes: tuple[TargetMode, ...] = ("位置", "朝向", "位姿")
        self._target_modes: dict[str, GuiDropdownHandle[TargetMode]] = {}
        # 每个末端的控件直接保留其模式；切换对象只改变显隐，不再维护第二份模式缓存。
        for name, end_effector in robot.end_effectors.items():
            self._target_modes[name] = gui.add_dropdown(
                "控制目标",
                options=modes,
                initial_value=mode_labels[end_effector.control_mode],
                visible=name == self.selected_object,
            )
        self._lock_base = gui.add_checkbox("固定基座", initial_value=True, visible=robot.floating)
        with gui.add_folder("关节锁定"):
            self._lock_joint_button = controls.button("锁定当前关节")
            self._unlock_joint_button = controls.button("解除当前关节")
            self._unlock_all_button = controls.button("解除全部关节")
        self._joint_sliders: dict[str, GuiSliderHandle[float]] = {}
        self._locked_joint_values: dict[str, GuiNumberHandle[float]] = {}
        with gui.add_folder("精确调整"):
            if robot.floating:
                with gui.add_folder("基座", expand_by_default=False):
                    self._root_position = gui.add_vector3("根位置 / m", initial_value=tuple(qpos[:3]), step=0.001)
                    self._root_rpy = gui.add_vector3("根 RPY / rad", initial_value=(0, 0, 0), step=0.01)
            groups = {group: names.copy() for group, names in robot.config.joint_groups.items()}
            assigned = {name for names in groups.values() for name in names}
            remaining = [name for name in robot.joint_names if name not in assigned]
            if remaining:
                groups.setdefault("其他关节", []).extend(remaining)
            for group, names in groups.items():
                if not names:
                    continue
                with gui.add_folder(group, expand_by_default=False):
                    for name in names:
                        i = robot.joint_names.index(name)
                        low = float(robot.lower[i]) if np.isfinite(robot.lower[i]) else -2 * np.pi
                        high = float(robot.upper[i]) if np.isfinite(robot.upper[i]) else 2 * np.pi
                        slide = robot.model.jnt_type[robot.joint_ids[i]] == mj.mjtJoint.mjJNT_SLIDE
                        handle = gui.add_slider(
                            name,
                            min=min(low, float(qpos[robot.qadr[i]])),
                            max=max(high, float(qpos[robot.qadr[i]])),
                            step=0.001,
                            initial_value=float(qpos[robot.qadr[i]]),
                            hint=f"{name} · {'移动关节 · m' if slide else '旋转关节 · rad'}",
                        )
                        self._joint_sliders[name] = handle
                        # Viser 滑块禁用时仍可编辑附带的数值框；锁定后用原生只读数值替换。
                        self._locked_joint_values[name] = gui.add_number(
                            name,
                            initial_value=float(qpos[robot.qadr[i]]),
                            step=0.001,
                            disabled=True,
                            visible=False,
                            hint=f"已锁定 · {'m' if slide else 'rad'}",
                        )
        self._reset_target_button = controls.button("重置目标", visible=False)
        self._home_button = controls.button("恢复初始姿态")

    def bind_actions(self, actions: PoseActions) -> None:
        controls = self._controls

        def prepare_joint(_client: ClientHandle, target: GuiButtonHandle) -> Callable[[], None]:
            action = actions.lock_joint if target is self._lock_joint_button else actions.unlock_joint
            return partial(action, self.selected_joint)

        for handle in (self._lock_joint_button, self._unlock_joint_button):
            handle.on_click(controls.client_action(prepare_joint))
        controls.click(self._unlock_all_button, actions.unlock_all_joints)
        controls.click(self._reset_target_button, actions.reset_target)
        controls.click(self._home_button, actions.home)
        for name, handle in self._joint_sliders.items():
            controls.input(handle, lambda value, name=name: actions.set_joint(name, value))
        if self._robot.floating:
            controls.input(self._root_position, actions.set_root_position)
            controls.input(self._root_rpy, actions.set_root_rpy)

    @property
    def selected_object(self) -> str | None:
        return self._object_labels.get(self._object_select.value)

    def select_object(self, name: str) -> None:
        self._object_select.value = next(label for label, value in self._object_labels.items() if value == name)

    @property
    def state(self) -> PoseState:
        selected = self.selected_object
        mode = self._target_modes.get(selected or "")
        return PoseState(selected, mode.value if mode is not None else None, self._lock_base.value)

    @property
    def selected_joint(self) -> str | None:
        selected = self.selected_object
        return selected.removeprefix("joint:") if selected is not None and selected.startswith("joint:") else None

    @property
    def track_position(self) -> bool:
        return self.state.mode != "朝向"

    @property
    def track_orientation(self) -> bool:
        return self.state.mode not in (None, "位置")

    @property
    def lock_base(self) -> bool:
        return self._lock_base.value

    def sync_pose(self, qpos: FloatArray, locked_joints: Collection[str]) -> None:
        robot = self._robot
        selected = self.selected_object
        is_end_effector = selected in robot.end_effectors
        self._lock_base.visible = is_end_effector and robot.floating
        for name, handle in self._target_modes.items():
            handle.visible = name == selected
        joint = self.selected_joint
        self._lock_joint_button.disabled = joint is None or joint in locked_joints
        self._unlock_joint_button.disabled = joint is None or joint not in locked_joints
        self._unlock_all_button.disabled = not locked_joints
        for name, handle in self._joint_sliders.items():
            self._controls.sync_slider(handle, float(qpos[robot.qadr[robot.joint_names.index(name)]]))
            locked = name in locked_joints
            handle.visible = not locked
            self._locked_joint_values[name].value = handle.value
            self._locked_joint_values[name].visible = locked
        if robot.floating:
            self._root_position.value = qpos[:3]
            self._root_rpy.value = rotation(qpos[3:7]).as_euler("xyz")

    def clear_drag_feedback(self) -> None:
        self._drag_feedback.content = ""
        self._drag_feedback.visible = False
        self._reset_target_button.visible = False

    def show_drag_result(self, result: DragResult) -> None:
        state = {"reached": "已到达", "projected": "跟随中", "blocked": "受限"}[result.state]
        errors: list[str] = []
        if self.track_position:
            errors.append(f"位置误差 {result.position_error * 1000:.2f} mm")
        if self.track_orientation:
            errors.append(f"方向误差 {result.orientation_error:.3f} rad")
        self._drag_feedback.content = f"**{state}** · " + " · ".join(errors)
        self._drag_feedback.visible = True
        self._reset_target_button.visible = True
