"""姿态编辑、场景拾取与拖拽跟随状态。"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import mujoco as mj
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation

from robot_motion_forge.robot.model import quaternion

from ..drag import JointTarget

if TYPE_CHECKING:
    from viser import ClientHandle

    from robot_motion_forge.application.editor import Editor
    from robot_motion_forge.motion.draft import Draft
    from robot_motion_forge.motion.types import PoseTarget

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
        self.dirty = True

    @property
    def has_target(self) -> bool:
        return self._goal is not None

    def sync(self, draft: Draft) -> None:
        self.sync_inputs()
        r, tab = self._editor.robot, self._tab
        self._view.update(draft.qpos, draft.locked_joints)
        tab.sync_pose(draft.qpos, draft.locked_joints)
        object_name = tab.selected_object
        if object_name is not None and object_name in r.end_effectors:
            position, quat = r.pose(self._view.data, object_name)
            self._drag.show(
                object_name,
                position,
                quat,
                track_position=tab.track_position,
                track_orientation=tab.track_orientation,
                target=self._goal,
            )
            self._view.set_control_hint(f"末端坐标系 · {object_name}")
        elif (name := tab.selected_joint) is not None and name not in draft.locked_joints:
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
        if state != self._input_state:
            self._input_state = state
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
        # 末端的原生点击也可能伴随场景点击，避免对同一选择重复重置拖拽目标。
        if object_name is not None and object_name != self._tab.selected_object:
            self.select_object(object_name)

    def lock_joint(self, name: str | None) -> None:
        if name is not None:
            self._editor.lock_joint(name)
        self.reset_target()

    def unlock_joint(self, name: str | None) -> None:
        if name is not None:
            self._editor.unlock_joint(name)
        self.reset_target()

    def unlock_all_joints(self) -> None:
        self._editor.unlock_all_joints()
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
            draft = self._editor.snapshot().draft
            self._editor.move_root(
                draft.qpos[:3] if position is None else position,
                draft.qpos[3:7] if rpy is None else quaternion(Rotation.from_euler("xyz", rpy)),
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
        if request is not None and request.end_effector == self._tab.selected_object:
            self._pause()
            self._goal = request
            self._following = True
        if self._goal is None or not self._following:
            return
        result = self._editor.drag_target(self._goal, lock_base=self._tab.lock_base)
        self.dirty = True
        self._tab.show_drag_result(result)
        if result.reached or not result.applied:
            self._following = False
