"""姿态编辑、场景拾取与拖拽跟随状态。"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import mujoco as mj
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation

from robot_motion_forge.rotations import rotation_to_wxyz

from ..drag import JointTarget

if TYPE_CHECKING:
    from viser import ClientHandle

    from robot_motion_forge.application.editor import Editor
    from robot_motion_forge.motion.constraints import EditConstraints
    from robot_motion_forge.motion.draft import Draft
    from robot_motion_forge.motion.types import PoseTarget
    from robot_motion_forge.robot.definition import ControlMode

    from ..drag import DragControls
    from ..notifications import Notifications
    from ..tabs.pose import PoseState, PoseTab
    from ..viewer import RobotView


class PoseController:
    def __init__(
        self,
        editor: Editor,
        tab: PoseTab,
        view: RobotView,
        drag: DragControls,
        *,
        pause: Callable[[], None],
        notifications: Notifications,
    ) -> None:
        self._editor, self._tab, self._view, self._drag = editor, tab, view, drag
        self._pause, self._notifications = pause, notifications
        self._goal: PoseTarget | None = None
        self._following = False
        self._client: ClientHandle | None = None
        self._gesture = 0
        self._input_state: PoseState | None = None
        self._constraint_state: EditConstraints | None = None
        self.dirty = True

    @property
    def has_target(self) -> bool:
        return self._goal is not None

    def sync(self, draft: Draft) -> None:
        self.sync_inputs()
        r, tab = self._editor.robot, self._tab
        self._view.update(draft.qpos, draft.constraints)
        tab.sync_pose(draft.qpos, draft.constraints)
        state = tab.state
        object_name = state.object_name
        if object_name is not None and object_name in r.frames and state.mode is not None:
            position, quat = r.pose(self._view.data, object_name)
            self._drag.show(
                object_name,
                position,
                quat,
                mode=state.mode,
                target=self._goal,
            )
            self._view.set_control_hint(f"控制坐标系 · {object_name}")
        elif (name := tab.selected_joint) is not None and name not in draft.constraints.locked_joints:
            index = r.joint_names.index(name)
            joint_position, joint_quat = r.joint_pose(self._view.data, name)
            self._drag.show_joint(
                name,
                float(draft.qpos[r.qadr[index]]),
                joint_position,
                joint_quat,
                slide=bool(r.model.jnt_type[r.joint_ids[index]] == mj.mjtJoint.mjJNT_SLIDE),
            )
            self._view.set_control_hint(f"关节 · {name}")
        else:
            self._drag.hide()
            self._view.set_control_hint("")
        self.dirty = False

    def sync_inputs(self) -> None:
        state = self._tab.state
        constraints = self._editor.constraints
        if state != self._input_state or constraints != self._constraint_state:
            self._input_state = state
            self._constraint_state = constraints
            self._editor.finish_edit()
            self.reset_target()

    def reset_target(self) -> None:
        self._goal = None
        self._following = False
        self._drag.clear()
        self._tab.clear_drag_feedback()
        self.dirty = True

    def select_object(self, object_name: str) -> None:
        self._tab.select_object(object_name)
        self.sync_inputs()

    def select_at(self, ray_origin: ArrayLike, ray_direction: ArrayLike) -> None:
        object_name = self._view.pick(ray_origin, ray_direction)
        # 坐标系的原生点击也可能伴随场景点击，避免对同一选择重复重置拖拽目标。
        if object_name is not None and object_name != self._tab.selected_object:
            self.select_object(object_name)

    def set_joint_locked(self, name: str | None, locked: bool) -> None:
        if name is not None:
            self._editor.set_joint_locked(name, locked)
        self.reset_target()

    def unlock_all_joints(self) -> None:
        self._editor.unlock_all_joints()
        self.reset_target()

    def set_anchor(self, name: str, mode: ControlMode | None) -> None:
        self._pause()
        self._editor.set_anchor(name, mode)
        self.reset_target()

    def clear_fixed_parts(self) -> None:
        self._editor.clear_fixed_parts()
        self.reset_target()

    def set_joint(self, name: str, value: float) -> None:
        self._pause()
        try:
            self._editor.set_joint(name, value)
        finally:
            self.reset_target()

    def set_root_position(self, value: tuple[float, float, float]) -> None:
        self._move_root(position=value)

    def set_root_rpy(self, value: tuple[float, float, float]) -> None:
        self._move_root(rpy=value)

    def _move_root(
        self, *, position: tuple[float, float, float] | None = None, rpy: tuple[float, float, float] | None = None
    ) -> None:
        self._pause()
        try:
            draft = self._editor.draft_snapshot()
            self._editor.move_root(
                draft.qpos[:3] if position is None else position,
                draft.qpos[3:7] if rpy is None else rotation_to_wxyz(Rotation.from_euler("xyz", rpy)),
            )
        finally:
            self.reset_target()

    def home(self) -> None:
        self._pause()
        self._editor.reset_pose()
        self.reset_target()

    def tick(self) -> None:
        try:
            self.sync_inputs()
            self._advance_drag()
        except Exception as exc:
            self._following = False
            self.dirty = True
            self._notifications.show("拖拽失败", str(exc), color="red", client=self._client)

    def _advance_drag(self) -> None:
        drag_input = self._drag.consume()
        self._client = drag_input.client
        if drag_input.gesture != self._gesture:
            self._editor.finish_edit()
            self._gesture = drag_input.gesture
            self._goal = None
            self._following = False
            self.dirty = True
        request = drag_input.target
        if isinstance(request, JointTarget):
            if request.name != self._tab.selected_joint:
                return
            self._pause()
            robot = self._editor.robot
            index = robot.joint_names.index(request.name)
            value = float(np.clip(request.value, robot.lower[index], robot.upper[index]))
            try:
                self._editor.set_joint(request.name, value)
            finally:
                self.dirty = True
            return
        if request is not None and request.frame == self._tab.selected_object:
            self._pause()
            self._goal = request
            self._following = True
        if self._goal is None or not self._following:
            return
        result = self._editor.drag_target(self._goal)
        self.dirty = True
        self._tab.show_drag_result(result)
        if result.reached or not result.applied:
            self._following = False
