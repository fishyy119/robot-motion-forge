"""姿态页：操作对象、拖拽反馈与精确调整。"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, NamedTuple, Protocol

import numpy as np

from robot_motion_forge.robot.definition import ControlMode
from robot_motion_forge.robot.model import Robot
from robot_motion_forge.utils.array_types import FloatArray
from robot_motion_forge.utils.rotations import rotation_from_wxyz

from ..controls import CONTROL_MODES, PoseSelection, grouped_joint_names
from ..viser_bridge import gui_uuid, json_for_script

if TYPE_CHECKING:
    from viser import GuiCheckboxHandle, GuiDropdownHandle, GuiNumberHandle, GuiSliderHandle

    from robot_motion_forge.motion.pose_constraints import EditConstraints
    from robot_motion_forge.solvers.drag import DragResult

    from ..controls import Controls

_ANCHOR_MODES: dict[str, ControlMode | None] = {
    "不固定": None,
    **{f"固定{label}": mode for label, mode in CONTROL_MODES.items()},
}


class PoseState(NamedTuple):
    selection: PoseSelection | None
    mode: ControlMode | None

    @property
    def frame_name(self) -> str | None:
        selected = self.selection
        return selected.name if selected is not None and selected.kind == "frame" else None

    @property
    def joint_name(self) -> str | None:
        selected = self.selection
        return selected.name if selected is not None and selected.kind == "joint" else None


class PoseActions(Protocol):
    def reset_target(self) -> None: ...
    def set_joint_locked(self, name: str, locked: bool) -> None: ...
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
        self._joint_indices = {name: index for index, name in enumerate(robot.joint_names)}
        self._object_labels = {f"部位 · {name}": PoseSelection("frame", name) for name in robot.frames}
        self._object_labels.update({f"关节 · {name}": PoseSelection("joint", name) for name in robot.joint_names})
        self._target_modes: dict[str, GuiDropdownHandle[str]] = {}
        self._joint_locks: dict[str, GuiCheckboxHandle] = {}
        with gui.add_folder("拖拽"):
            self._object_select = gui.add_dropdown("操作对象", options=tuple(self._object_labels) or ("—",))
            state = self.state
            # 每个坐标系的控件直接保留其模式；切换对象只改变显隐，不再维护第二份模式缓存。
            for name, frame in robot.frames.items():
                self._target_modes[name] = gui.add_dropdown(
                    "控制分量",
                    options=tuple(CONTROL_MODES),
                    initial_value=next(label for label, mode in CONTROL_MODES.items() if mode == frame.control_mode),
                    visible=name == state.frame_name,
                )
            for name in robot.joint_names:
                self._joint_locks[name] = gui.add_checkbox(
                    "锁定关节", initial_value=False, visible=name == state.joint_name
                )
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
            self._locked_joint_list = gui.add_markdown("")
            self._unlock_all_button = controls.button("解除全部关节")
        self._joint_sliders: dict[str, GuiSliderHandle[float]] = {}
        self._locked_joint_values: dict[str, GuiNumberHandle[float]] = {}
        with gui.add_folder("精确调整", expand_by_default=False):
            if robot.floating:
                with gui.add_folder("torso", expand_by_default=False):
                    self._root_position = gui.add_vector3("位置 / m", initial_value=tuple(qpos[:3]), step=0.001)
                    self._root_rpy = gui.add_vector3("RPY / rad", initial_value=(0, 0, 0), step=0.01)
            for group, names in grouped_joint_names(robot).items():
                with gui.add_folder(group, expand_by_default=False):
                    for name in names:
                        i = self._joint_indices[name]
                        low = float(robot.lower[i]) if np.isfinite(robot.lower[i]) else -2 * np.pi
                        high = float(robot.upper[i]) if np.isfinite(robot.upper[i]) else 2 * np.pi
                        slide = robot.is_slide_joint(i)
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

    def scene_menu_metadata(
        self, frame_paths: dict[str, str], body_joints: dict[str, tuple[str, ...]], *, gizmo_path: str
    ) -> str:
        # 场景菜单提交到对应的侧栏控件，沿用姿态编辑的状态同步。
        frames = {
            frame_paths[name]: {
                "name": name,
                "dropdowns": {"固定模式": gui_uuid(handle), "控制模式": gui_uuid(self._target_modes[name])},
            }
            for name, handle in self._anchor_modes.items()
        }
        checkboxes = {name: gui_uuid(handle) for name, handle in self._joint_locks.items()}
        data = json_for_script(
            {"frames": frames, "joints": checkboxes, "bodies": body_joints, "gizmo": gizmo_path}, ensure_ascii=False
        )
        return f'<script type="application/json" data-rmf-scene-menu>{data}</script>'

    def bind_actions(self, actions: PoseActions) -> None:
        controls = self._controls
        for name, handle in self._joint_locks.items():
            controls.input(handle, partial(actions.set_joint_locked, name))
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
    def selected_object(self) -> PoseSelection | None:
        return self._object_labels.get(self._object_select.value)

    def select_object(self, selection: PoseSelection) -> None:
        self._object_select.value = next(label for label, value in self._object_labels.items() if value == selection)

    @property
    def state(self) -> PoseState:
        selected = self.selected_object
        mode = self._target_modes.get(selected.name) if selected is not None and selected.kind == "frame" else None
        return PoseState(selected, CONTROL_MODES[mode.value] if mode is not None else None)

    def sync_pose(self, qpos: FloatArray, constraints: EditConstraints, state: PoseState) -> None:
        robot = self._robot
        frame = state.frame_name
        locked_joints = constraints.locked_joints
        for name, handle in self._anchor_modes.items():
            anchor = constraints.anchors.get(name)
            mode = anchor.mode if anchor else None
            handle.value = next(label for label, value in _ANCHOR_MODES.items() if value == mode)
        self._clear_fixed_parts_button.disabled = not constraints.anchors
        for name, handle in self._target_modes.items():
            handle.visible = name == frame
        joint = state.joint_name
        self._unlock_all_button.disabled = not locked_joints
        other_locked_joints = [name for name in locked_joints if name != joint]
        self._joint_locks_folder.visible = bool(other_locked_joints)
        self._locked_joint_list.content = "\n".join(f"- `{name}`" for name in other_locked_joints)
        for name, handle in self._joint_sliders.items():
            self._controls.sync_slider(handle, float(qpos[robot.qadr[self._joint_indices[name]]]))
            locked = name in locked_joints
            lock = self._joint_locks[name]
            lock.value, lock.visible = locked, name == joint
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

    def show_drag_result(self, result: DragResult, *, mode: ControlMode | None) -> None:
        state = {"reached": "已到达", "projected": "跟随中", "blocked": "受限"}[result.state]
        errors: list[str] = []
        if mode in ("position", "pose"):
            errors.append(f"位置误差 {result.position_error * 1000:.2f} mm")
        if mode in ("orientation", "pose"):
            errors.append(f"方向误差 {result.orientation_error:.3f} rad")
        self._drag_feedback.content = f"**{state}** · " + " · ".join(errors)
        self._drag_feedback.visible = True
        self._reset_target_button.visible = True
