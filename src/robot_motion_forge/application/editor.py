"""编辑会话：撤销、草稿、关键帧与求解缓存；无 Viser 依赖。"""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

import numpy as np
from numpy.typing import ArrayLike

from robot_motion_forge.io.project import load_project, save_project
from robot_motion_forge.motion.constraints import EditConstraints
from robot_motion_forge.motion.draft import Draft, PoseSource
from robot_motion_forge.motion.project import Project
from robot_motion_forge.motion.trajectory import ReferenceTrajectory, SolvedTrajectory, validate_trajectory
from robot_motion_forge.motion.types import FrameTask, Keyframe, PoseTarget, SolverSettings
from robot_motion_forge.robot.definition import ControlMode
from robot_motion_forge.robot.model import FloatArray, Robot
from robot_motion_forge.solvers.drag import DragResult, follow_target
from robot_motion_forge.solvers.ik import IKResult, solve_ik


@dataclass(frozen=True)
class _EditState:
    project: Project
    draft: Draft
    solve_fingerprint: str


class Editor:
    def __init__(
        self,
        robot: Robot,
        project: Project | None = None,
        trajectory: SolvedTrajectory | None = None,
        *,
        draft: Draft | None = None,
    ) -> None:
        self.robot = robot
        project = (project or Project.new(robot)).normalized(robot)
        initial_draft = deepcopy(draft) if draft is not None else Draft.initial(robot)
        initial_draft.validate(robot, project)
        self._state = _EditState(project, initial_draft, project.solve_fingerprint())
        if trajectory is not None:
            validate_trajectory(robot, project.duration, trajectory)
        self._trajectory = trajectory
        self._reference: ReferenceTrajectory | None = None
        self._undo: list[_EditState] = []
        self._redo: list[_EditState] = []
        self._merge_key: tuple[str, str] | None = None
        self._last_edit_time = 0.0
        self.revision = 0
        self.project_revision = 0
        self.draft_revision = 0
        if draft is None:
            self.scrub(initial_draft.cursor)

    @classmethod
    def load(cls, robot: Robot, path: Path) -> "Editor":
        project, draft, trajectory = load_project(path, robot)
        return cls(robot, project, trajectory, draft=draft)

    def project_snapshot(self) -> Project:
        return self._state.project.model_copy(deep=True)

    def draft_snapshot(self) -> Draft:
        return deepcopy(self._state.draft)

    @property
    def duration(self) -> float:
        return self._state.project.duration

    @property
    def solve_fingerprint(self) -> str:
        return self._state.solve_fingerprint

    @property
    def trajectory(self) -> SolvedTrajectory | None:
        return self._trajectory

    @property
    def constraints(self) -> EditConstraints:
        return self._state.draft.constraints.model_copy(deep=True)

    def edit(self, operation: Callable[[Project, Draft], None], *, merge_key: tuple[str, str] | None = None) -> None:
        # 已提交工程只读；历史保留旧状态，候选修改在独立副本上执行。
        old = self._state
        candidate, draft = deepcopy(old.project), deepcopy(old.draft)
        operation(candidate, draft)
        if candidate != old.project:
            candidate = candidate.normalized(self.robot)
            fingerprint = candidate.solve_fingerprint()
        else:
            candidate, fingerprint = old.project, old.solve_fingerprint
        self._commit_edit(_EditState(candidate, draft, fingerprint), merge_key)

    def _edit_draft(self, operation: Callable[[Draft], None], *, merge_key: tuple[str, str] | None = None) -> None:
        old = self._state
        draft = deepcopy(old.draft)
        operation(draft)
        self._commit_edit(_EditState(old.project, draft, old.solve_fingerprint), merge_key)

    def _commit_edit(self, state: _EditState, merge_key: tuple[str, str] | None) -> None:
        old = self._state
        state.draft.validate(self.robot, state.project)
        if (state.project is old.project or state.project == old.project) and state.draft.matches(old.draft):
            return
        now = monotonic()
        # 连续拖拽、滑块与文字编辑合并撤销；停顿或切换操作后开启新记录。
        if merge_key is None or merge_key != self._merge_key or now - self._last_edit_time > 0.5:
            self._undo.append(old)
            self._undo = self._undo[-100:]
        self._merge_key, self._last_edit_time = merge_key, now
        self._redo.clear()
        self._commit(state)

    def finish_edit(self) -> None:
        self._merge_key = None

    def _commit(self, state: _EditState) -> None:
        previous, self._state = self._state, state
        self.revision += 1
        if previous.project is not state.project and previous.project != state.project:
            self.project_revision += 1
            old, new = previous.project, state.project
            if (
                old.duration != new.duration
                or old.settings.terminal_condition != new.settings.terminal_condition
                or old.keyframes != new.keyframes
            ):
                self._reference = None
        if not previous.draft.matches(state.draft):
            self.draft_revision += 1
        if previous.solve_fingerprint != state.solve_fingerprint:
            self._trajectory = None

    def _draft_changed(self) -> None:
        self.draft_revision += 1
        self.revision += 1

    def undo(self) -> None:
        if self._undo:
            self._redo.append(self._state)
            self.finish_edit()
            self._commit(self._undo.pop())

    def redo(self) -> None:
        if self._redo:
            self._undo.append(self._state)
            self.finish_edit()
            self._commit(self._redo.pop())

    def install_solution(self, trajectory: SolvedTrajectory, fingerprint: str) -> bool:
        if fingerprint != self._state.solve_fingerprint:
            return False
        validate_trajectory(self.robot, self._state.project.duration, trajectory)
        self._trajectory = trajectory
        self.revision += 1
        return True

    def reference_trajectory(self) -> ReferenceTrajectory:
        if self._reference is None:
            self._reference = self._state.project.keyframe_trajectory(self.robot, preview=True)
        return self._reference

    def seek(self, time: float) -> None:
        """定位到连续轨迹，供播放与轨迹取姿使用。"""
        time = float(np.clip(time, 0, self._state.project.duration))
        trajectory = self.trajectory or self.reference_trajectory()
        self._move_cursor(time, trajectory.sample(self.robot, [time])[0])

    def _move_cursor(
        self, time: float, qpos: FloatArray | None, *, selected: str | None = None, held: str | None = None
    ) -> None:
        self.finish_edit()
        draft = self._state.draft
        draft.cursor, draft.selected_key, draft.held_key = time, selected, held
        if qpos is not None:
            draft.qpos = qpos
            draft.constraints.reanchor(self.robot, qpos)
        self._draft_changed()

    def advance(self, elapsed: float) -> bool:
        """按工程终端条件推进游标并采样，返回是否继续播放。"""
        project = self._state.project
        time = self._state.draft.cursor + elapsed
        if project.settings.terminal_condition == "loop":
            time %= project.duration
        self.seek(time)
        return time < project.duration

    @property
    def pose_source(self) -> PoseSource:
        return self._state.draft.pose_source

    def set_pose_source(self, source: PoseSource) -> None:
        draft = self._state.draft
        if source != draft.pose_source:
            draft.pose_source = source
            draft.held_key = None
            self.scrub(draft.cursor)

    def _previous_key(self, time: float) -> Keyframe:
        keys = self._state.project.keyframes
        return max(
            (key for key in keys if key.time <= time),
            key=lambda key: key.time,
            default=min(keys, key=lambda key: key.time),
        )

    def scrub(self, time: float, *, snap_tolerance: float = 0.0) -> None:
        time = float(np.clip(time, 0, self._state.project.duration))
        key = min(self._state.project.keyframes, key=lambda key: (abs(key.time - time), key.time))
        # 滑块按步长取整，命中后吸附到关键帧的准确时刻；轨迹取样始终作为独立草稿。
        snapped = abs(key.time - time) <= snap_tolerance + 1e-8
        if self.pose_source == "trajectory":
            self.seek(key.time if snapped else time)
            return
        if snapped:
            self.select_key(key.id)
            return
        previous = self._previous_key(time)
        qpos = None if self._state.draft.held_key == previous.id else np.array(previous.qpos)
        self._move_cursor(time, qpos, held=previous.id)

    def set_timing(self, duration: float | None = None, fps: float | None = None) -> None:
        def operation(project: Project, draft: Draft) -> None:
            if duration is not None:
                project.duration = duration
            if fps is not None:
                project.fps = fps
            draft.cursor = min(draft.cursor, project.duration)

        field = "duration" if duration is not None else "fps"
        self.edit(operation, merge_key=("timing", field) if duration is None or fps is None else None)

    def set_settings(self, **values: float | int | str) -> None:
        settings = SolverSettings.model_validate({**self._state.project.settings.model_dump(), **values})
        merge_key = (
            ("settings", next(iter(values)))
            if len(values) == 1 and not isinstance(next(iter(values.values())), (bool, str))
            else None
        )
        self.edit(lambda project, draft: setattr(project, "settings", settings), merge_key=merge_key)

    def set_frame_task(self, name: str, task: FrameTask) -> None:
        current = self._state.project.frame_tasks.get(name)

        def operation(project: Project, draft: Draft) -> None:
            project.frame_tasks[name] = task.model_copy(deep=True)

        # 模式和参考系切换各自撤销；同一任务的连续权重调整合并。
        merge_key = (
            ("frame_task", name)
            if current is not None and (current.mode, current.reference_body) == (task.mode, task.reference_body)
            else None
        )
        self.edit(operation, merge_key=merge_key)

    def capture(self, name: str, *, locked: bool = True) -> None:
        key = Keyframe(
            time=self._state.draft.cursor,
            qpos=self._state.draft.qpos.tolist(),
            name=name,
            locked=locked,
        )

        def operation(project: Project, draft: Draft) -> None:
            project.keyframes.append(key)
            draft.selected_key = key.id
            draft.held_key = key.id

        self.edit(operation)
        self.finish_edit()

    def select_key(self, key_id: str | None) -> None:
        if key_id is None:
            self.finish_edit()
            if self._state.draft.selected_key is not None:
                self._state.draft.selected_key = None
                self._draft_changed()
            return
        key = next(k for k in self._state.project.keyframes if k.id == key_id)
        self._move_cursor(key.time, np.array(key.qpos), selected=key.id, held=key.id)

    def update_key(
        self,
        key_id: str,
        *,
        time: float | None = None,
        name: str | None = None,
        locked: bool | None = None,
        zero_velocity: bool | None = None,
    ) -> None:
        def operation(project: Project, draft: Draft) -> None:
            key = next((k for k in project.keyframes if k.id == key_id), None)
            if key is None:
                raise ValueError("关键帧已不存在")
            if time is not None:
                key.time = time
                if draft.selected_key == key_id:
                    draft.cursor = time
            if name is not None:
                key.name = name
            if locked is not None:
                key.locked = locked
            if zero_velocity is not None:
                key.zero_velocity = zero_velocity

        updates = {"time": time, "name": name, "locked": locked, "zero_velocity": zero_velocity}
        fields = [field for field, value in updates.items() if value is not None]
        merge_key = (fields[0], key_id) if len(fields) == 1 and fields[0] in ("time", "name") else None
        self.edit(operation, merge_key=merge_key)

    def delete_key(self, key_id: str) -> None:
        def operation(project: Project, draft: Draft) -> None:
            project.keyframes = [key for key in project.keyframes if key.id != key_id]
            if draft.selected_key == key_id:
                draft.selected_key = None
            if draft.held_key == key_id:
                draft.held_key = None

        self.edit(operation)

    def _set_pose(self, qpos: FloatArray, *, merge: bool = True, constraints: EditConstraints | None = None) -> None:
        current = self._state.draft
        selected = current.selected_key
        if selected is not None:
            key = next(key for key in self._state.project.keyframes if key.id == selected)
            if abs(key.time - current.cursor) > 1e-8:
                selected = None

        def update_draft(draft: Draft) -> None:
            draft.selected_key = selected
            draft.qpos = qpos.copy()
            if draft.pose_source == "previous_keyframe" and draft.held_key is None:
                draft.held_key = self._previous_key(draft.cursor).id
            if constraints is not None:
                draft.constraints = constraints.model_copy(deep=True)

        merge_key = ("pose", current.selected_key or "draft") if merge else None
        if selected is None:
            self._edit_draft(update_draft, merge_key=merge_key)
            return

        def update_key(project: Project, draft: Draft) -> None:
            key = next(key for key in project.keyframes if key.id == selected)
            key.qpos = qpos.tolist()
            update_draft(draft)

        self.edit(update_key, merge_key=merge_key)

    def set_joint_locked(self, name: str, locked: bool) -> None:
        if name not in self.robot.joint_names:
            raise ValueError(f"未知关节：{name}")

        def operation(draft: Draft) -> None:
            joints = draft.constraints.locked_joints
            if locked and name not in joints:
                joints.append(name)
            elif not locked and name in joints:
                joints.remove(name)

        self._edit_draft(operation)

    def unlock_all_joints(self) -> None:
        self._edit_draft(lambda draft: draft.constraints.locked_joints.clear())

    def set_anchor(self, name: str, mode: ControlMode | None) -> None:
        self.robot.frame(name)

        def operation(draft: Draft) -> None:
            if mode is None:
                draft.constraints.anchors.pop(name, None)
            elif name not in draft.constraints.anchors or draft.constraints.anchors[name].mode != mode:
                draft.constraints.capture(self.robot, self.robot.forward(draft.qpos), name, mode)

        self._edit_draft(operation)

    def clear_fixed_parts(self) -> None:
        self._edit_draft(lambda draft: draft.constraints.anchors.clear())

    def reset_pose(self) -> None:
        initial = Draft.initial(self.robot)
        self._set_pose(initial.qpos, merge=False, constraints=initial.constraints)

    def set_joint(self, name: str, value: float) -> None:
        if name in self._state.draft.constraints.locked_joints:
            raise ValueError(f"关节已锁定：{name}")
        candidate = self._state.draft.qpos.copy()
        candidate[self.robot.qadr[self.robot.joint_names.index(name)]] = value
        self._set_pose(self.robot.validate_qpos(candidate))

    def move_target(self, target: PoseTarget) -> IKResult:
        result = solve_ik(
            self.robot,
            self._state.draft.qpos,
            [target],
            constraints=self._state.draft.constraints,
        )
        # IK 失败不提交部分姿态；用户可以放松姿态方向或移动 torso 后重试。
        if result.success:
            self._set_pose(result.qpos)
        return result

    def move_root(self, position: ArrayLike, quaternion: ArrayLike) -> None:
        if not self.robot.floating:
            raise ValueError("当前模型的 torso 没有可编辑的自由度")
        if np.shape(position) != (3,) or np.shape(quaternion) != (4,):
            raise ValueError("torso 位置需要三个分量，朝向需要四个 wxyz 四元数分量")
        candidate = self._state.draft.qpos.copy()
        candidate[:3], candidate[3:7] = position, quaternion
        self._set_pose(self.robot.validate_qpos(candidate))

    def drag_target(
        self,
        target: PoseTarget,
        *,
        position_step: float = 0.003,
        rotation_step: float = np.pi / 60,
    ) -> DragResult:
        result = follow_target(
            self.robot,
            self._state.draft.qpos,
            target,
            constraints=self._state.draft.constraints,
            position_step=position_step,
            rotation_step=rotation_step,
        )
        if result.applied:
            self._set_pose(result.qpos)
        return result

    def save(self, path: Path) -> None:
        save_project(
            path,
            self.robot,
            self._state.project,
            self._state.draft,
            self.trajectory,
        )
