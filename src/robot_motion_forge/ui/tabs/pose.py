"""姿态页：操作对象、拖拽反馈与精确调整。"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, NamedTuple, Protocol

import mujoco as mj
import numpy as np

from robot_motion_forge.robot.definition import ControlMode
from robot_motion_forge.robot.model import FloatArray, Robot
from robot_motion_forge.rotations import rotation_from_wxyz

from ..controls import CONTROL_MODES

if TYPE_CHECKING:
    from viser import ClientHandle, GuiCheckboxHandle, GuiDropdownHandle, GuiNumberHandle, GuiSliderHandle

    from robot_motion_forge.motion.constraints import EditConstraints
    from robot_motion_forge.solvers.drag import DragResult

    from ..controls import Controls

_ANCHOR_MODES: dict[str, ControlMode | None] = {
    "不固定": None,
    **{f"固定{label}": mode for label, mode in CONTROL_MODES.items()},
}


class PoseState(NamedTuple):
    object_name: str | None
    mode: ControlMode | None


class PoseActions(Protocol):
    def reset_target(self) -> None: ...
    def set_joint_locked(self, name: str | None, locked: bool) -> None: ...
    def unlock_all_joints(self) -> None: ...
    def set_anchor(self, name: str, mode: ControlMode | None) -> None: ...
    def clear_fixed_parts(self) -> None: ...
    def home(self) -> None: ...
    def set_joint(self, name: str, value: float) -> None: ...
    def set_root_position(self, value: tuple[float, float, float]) -> None: ...
    def set_root_rpy(self, value: tuple[float, float, float]) -> None: ...


class PoseTab:
    def __init__(self, controls: Controls, *, robot: Robot, qpos: FloatArray) -> None:
        gui = controls.gui
        self._controls = controls
        self._robot = robot
        self._object_labels = {f"部位 · {name}": name for name in robot.frames}
        self._object_labels.update({f"关节 · {name}": f"joint:{name}" for name in robot.joint_names})
        self._target_modes: dict[str, GuiDropdownHandle[str]] = {}
        with gui.add_folder("拖拽"):
            self._object_select = gui.add_dropdown("操作对象", options=tuple(self._object_labels) or ("—",))
            # 每个坐标系的控件直接保留其模式；切换对象只改变显隐，不再维护第二份模式缓存。
            for name, frame in robot.frames.items():
                self._target_modes[name] = gui.add_dropdown(
                    "控制分量",
                    options=tuple(CONTROL_MODES),
                    initial_value=next(label for label, mode in CONTROL_MODES.items() if mode == frame.control_mode),
                    visible=name == self.selected_object,
                )
            self._lock_joint = gui.add_checkbox("锁定关节", initial_value=False, visible=False)
            self._drag_feedback = gui.add_markdown("", visible=False)
            self._reset_target_button = controls.button("重置目标", visible=False)
        self._anchor_modes: dict[str, GuiDropdownHandle[str]] = {}
        with gui.add_folder("固定部位"):
            for name in robot.frames:
                self._anchor_modes[name] = gui.add_dropdown(
                    name,
                    options=tuple(_ANCHOR_MODES),
                    initial_value="不固定",
                )
            self._clear_fixed_parts_button = controls.button("解除全部部位固定", disabled=True)
        with gui.add_folder("已锁定关节", expand_by_default=False) as self._joint_locks_folder:
            self._joint_locks = gui.add_markdown("")
            self._unlock_all_button = controls.button("解除全部关节")
        self._joint_sliders: dict[str, GuiSliderHandle[float]] = {}
        self._locked_joint_values: dict[str, GuiNumberHandle[float]] = {}
        with gui.add_folder("精确调整", expand_by_default=False):
            if robot.floating:
                with gui.add_folder("torso", expand_by_default=False):
                    self._root_position = gui.add_vector3("位置 / m", initial_value=tuple(qpos[:3]), step=0.001)
                    self._root_rpy = gui.add_vector3("RPY / rad", initial_value=(0, 0, 0), step=0.01)
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
        self._home_button = controls.button("恢复初始姿态")

    def bind_actions(self, actions: PoseActions) -> None:
        controls = self._controls

        def prepare_joint(_client: ClientHandle, target: GuiCheckboxHandle) -> Callable[[], None]:
            return partial(actions.set_joint_locked, self.selected_joint, target.value)

        self._lock_joint.on_update(controls.client_action(prepare_joint))
        controls.click(self._unlock_all_button, actions.unlock_all_joints)
        for name, handle in self._anchor_modes.items():
            controls.input(handle, lambda value, name=name: actions.set_anchor(name, _ANCHOR_MODES[value]))
        controls.click(self._clear_fixed_parts_button, actions.clear_fixed_parts)
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
        return PoseState(selected, CONTROL_MODES[mode.value] if mode is not None else None)

    @property
    def selected_joint(self) -> str | None:
        selected = self.selected_object
        return selected.removeprefix("joint:") if selected is not None and selected.startswith("joint:") else None

    def sync_pose(self, qpos: FloatArray, constraints: EditConstraints) -> None:
        robot = self._robot
        selected = self.selected_object
        locked_joints = constraints.locked_joints
        for name, handle in self._anchor_modes.items():
            anchor = constraints.anchors.get(name)
            mode = anchor.mode if anchor else None
            handle.value = next(label for label, value in _ANCHOR_MODES.items() if value == mode)
        self._clear_fixed_parts_button.disabled = not constraints.anchors
        for name, handle in self._target_modes.items():
            handle.visible = name == selected
        joint = self.selected_joint
        self._lock_joint.visible = joint is not None
        self._lock_joint.value = joint in locked_joints
        self._unlock_all_button.disabled = not locked_joints
        other_locked_joints = [name for name in locked_joints if name != joint]
        self._joint_locks_folder.visible = bool(other_locked_joints)
        self._joint_locks.content = "\n".join(f"- `{name}`" for name in other_locked_joints)
        for name, handle in self._joint_sliders.items():
            self._controls.sync_slider(handle, float(qpos[robot.qadr[robot.joint_names.index(name)]]))
            locked = name in locked_joints
            handle.visible = not locked
            self._locked_joint_values[name].value = handle.value
            self._locked_joint_values[name].visible = locked
        if robot.floating:
            self._root_position.value = qpos[:3]
            self._root_rpy.value = rotation_from_wxyz(qpos[3:7]).as_euler("xyz")

    def clear_drag_feedback(self) -> None:
        self._drag_feedback.content = ""
        self._drag_feedback.visible = False
        self._reset_target_button.visible = False

    def show_drag_result(self, result: DragResult) -> None:
        state = {"reached": "已到达", "projected": "跟随中", "blocked": "受限"}[result.state]
        errors: list[str] = []
        if self.state.mode in ("position", "pose"):
            errors.append(f"位置误差 {result.position_error * 1000:.2f} mm")
        if self.state.mode in ("orientation", "pose"):
            errors.append(f"方向误差 {result.orientation_error:.3f} rad")
        self._drag_feedback.content = f"**{state}** · " + " · ".join(errors)
        self._drag_feedback.visible = True
        self._reset_target_button.visible = True
