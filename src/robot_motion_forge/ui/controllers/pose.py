"""姿态编辑、场景拾取与拖拽跟随状态。"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation

from robot_motion_forge.utils.rotations import rotation_to_wxyz

from ..scene.gizmo import JointTarget

if TYPE_CHECKING:
    from viser import ClientHandle

    from robot_motion_forge.editor import Editor
    from robot_motion_forge.motion.draft import Draft
    from robot_motion_forge.motion.pose_constraints import EditConstraints, PoseTarget
    from robot_motion_forge.robot.definition import ControlMode

    from ..notifications import Notifications
    from ..scene.gizmo import DragControls
    from ..scene.robot_view import RobotView
    from ..tabs.pose import PoseState, PoseTab


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

    def sync(self, draft: Draft, state: PoseState) -> None:
        r, tab = self._editor.robot, self._tab
        tab.sync_pose(draft.qpos, draft.constraints, state)
        frame = state.frame_name
        if frame is not None and state.mode is not None:
            position, quat = r.frame_pose(self._view.data, frame)
            self._drag.show(
                frame,
                position,
                quat,
                mode=state.mode,
                target=self._goal,
            )
            hint = f"控制坐标系 · {frame}"
        elif (name := state.joint_name) is not None and name not in draft.constraints.locked_joints:
            index = r.joint_names.index(name)
            self._drag.show_joint(r, self._view.data, index)
            hint = f"关节 · {name}"
        else:
            self._drag.hide()
            hint = ""
        self._view.set_control_hint(self._drag.gizmo.name, hint)
        self.dirty = False

    def sync_inputs(self, draft: Draft) -> PoseState:
        state = self._tab.state
        constraints = draft.constraints
        previous = self._constraint_state
        self._constraint_state = constraints
        if state != self._input_state or (constraints is not previous and constraints != previous):
            self._input_state = state
            self._editor.finish_edit()
            self.reset_target()
        return state

    def reset_target(self) -> None:
        self._goal = None
        self._following = False
        self._drag.clear()
        self._tab.clear_drag_feedback()
        self.dirty = True

    def select_at(self, ray_origin: ArrayLike, ray_direction: ArrayLike) -> None:
        selection = self._view.pick(ray_origin, ray_direction)
        # 重复点击同一对象保留现有拖拽目标。
        if selection != self._tab.selected_object:
            self._tab.select_object(selection)

    def set_joint_locked(self, name: str, locked: bool) -> None:
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

    def tick(self, draft: Draft) -> None:
        try:
            state = self.sync_inputs(draft)
            self._advance_drag(state)
        except Exception as exc:
            self._following = False
            self.dirty = True
            self._notifications.show("拖拽失败", str(exc), color="red", client=self._client)

    def _advance_drag(self, state: PoseState) -> None:
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
            if request.name != state.joint_name:
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
        if request is not None and request.frame == state.frame_name:
            self._pause()
            self._goal = request
            self._following = True
        if self._goal is None or not self._following:
            return
        result = self._editor.drag_target(self._goal)
        self.dirty = True
        self._tab.show_drag_result(result, mode=state.mode)
        if result.reached or not result.applied:
            self._following = False
