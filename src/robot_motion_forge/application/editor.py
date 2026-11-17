"""编辑会话：撤销、草稿、关键帧与求解缓存；无 Viser 依赖。"""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

import numpy as np
from numpy.typing import ArrayLike

from robot_motion_forge.io.project import save_project
from robot_motion_forge.motion.constraints import EditConstraints
from robot_motion_forge.motion.draft import Draft
from robot_motion_forge.motion.project import Project
from robot_motion_forge.motion.trajectory import Trajectory, validate_trajectory
from robot_motion_forge.motion.types import FrameTask, Keyframe, PoseTarget, SolverSettings
from robot_motion_forge.robot.definition import ControlMode
from robot_motion_forge.robot.model import FloatArray, Robot
from robot_motion_forge.solvers.drag import DragResult, follow_target
from robot_motion_forge.solvers.ik import IKResult, solve_ik


@dataclass(frozen=True)
class EditorSnapshot:
    project: Project
    draft: Draft
    solve_fingerprint: str


class Editor:
    def __init__(
        self,
        robot: Robot,
        project: Project | None = None,
        trajectory: Trajectory | None = None,
        *,
        draft: Draft | None = None,
    ) -> None:
        self.robot = robot
        project = (project or Project.new(robot)).validated(robot)
        initial_draft = deepcopy(draft) if draft is not None else Draft.initial(robot)
        initial_draft.validate(robot, project)
        self._state = EditorSnapshot(project, initial_draft, project.solve_fingerprint())
        if trajectory is not None:
            validate_trajectory(robot, project, trajectory)
        self.trajectory = trajectory
        self._preview: Trajectory | None = None
        self._undo: list[EditorSnapshot] = []
        self._redo: list[EditorSnapshot] = []
        self._merge_key: tuple[str, str] | None = None
        self._last_edit_time = 0.0
        self.revision = 0
        self.project_revision = 0
        self.trajectory_revision = 0
        if draft is None:
            self.scrub(initial_draft.cursor)

    def snapshot(self) -> EditorSnapshot:
        """返回独立快照；会话修改统一通过编辑命令提交。"""
        return deepcopy(self._state)

    @property
    def constraints(self) -> EditConstraints:
        return self._state.draft.constraints.model_copy(deep=True)

    def edit(self, operation: Callable[[Project, Draft], None], *, merge_key: tuple[str, str] | None = None) -> None:
        # 已提交工程只读；历史保留旧状态，候选修改在独立副本上执行。
        old = self._state
        candidate, draft = deepcopy(old.project), deepcopy(old.draft)
        operation(candidate, draft)
        if candidate != old.project:
            candidate = candidate.validated(self.robot)
            fingerprint = candidate.solve_fingerprint()
        else:
            candidate, fingerprint = old.project, old.solve_fingerprint
        draft.validate(self.robot, candidate)
        if candidate == old.project and draft.matches(old.draft):
            return
        now = monotonic()
        # 连续拖拽、滑块与文字编辑合并撤销；停顿或切换操作后开启新记录。
        if merge_key is None or merge_key != self._merge_key or now - self._last_edit_time > 0.5:
            self._undo.append(old)
            self._undo = self._undo[-100:]
        self._merge_key, self._last_edit_time = merge_key, now
        self._redo.clear()
        self._commit(EditorSnapshot(candidate, draft, fingerprint))

    def finish_edit(self) -> None:
        self._merge_key = None

    def _commit(self, state: EditorSnapshot) -> None:
        previous, self._state = self._state, state
        self.revision += 1
        if previous.project != state.project:
            self.project_revision += 1
        if previous.solve_fingerprint != state.solve_fingerprint:
            self.trajectory = None
            self._preview = None
            self.trajectory_revision += 1

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

    def install_solution(self, trajectory: Trajectory, fingerprint: str) -> bool:
        if fingerprint != self._state.solve_fingerprint:
            return False
        validate_trajectory(self.robot, self._state.project, trajectory)
        self.trajectory = trajectory
        self.trajectory_revision += 1
        self.revision += 1
        return True

    def preview(self) -> Trajectory:
        if self.trajectory is not None:
            return self.trajectory
        if self._preview is None:
            # 预览插值也保留非帧对齐关键帧，避免在二次插值中损失手工姿态。
            project = self._state.project
            keyframes = project.keyframe_trajectory()
            times = np.unique(np.r_[0.0, project.duration, keyframes.times])
            self._preview = Trajectory(times, keyframes.sample(self.robot, times))
        return self._preview

    def seek(self, time: float) -> None:
        self.select_key(None)
        draft = self._state.draft
        draft.cursor = float(np.clip(time, 0, self._state.project.duration))
        draft.qpos = self.preview().sample(self.robot, [draft.cursor])[0]
        draft.constraints.reanchor(self.robot, draft.qpos)
        self.revision += 1

    def advance(self, elapsed: float) -> None:
        self.seek((self._state.draft.cursor + elapsed) % self._state.project.duration)

    def scrub(self, time: float, *, snap_tolerance: float = 0.0) -> None:
        time = float(np.clip(time, 0, self._state.project.duration))
        key = min(self._state.project.keyframes, key=lambda key: (abs(key.time - time), key.time))
        # 滑块按步长取整，命中后回到关键帧的准确时刻与作者姿态。
        if abs(key.time - time) <= snap_tolerance + 1e-8:
            self.select_key(key.id)
        else:
            self.seek(time)

    def set_timing(self, duration: float | None = None, fps: float | None = None) -> None:
        def operation(project: Project, draft: Draft) -> None:
            if duration is not None:
                project.duration = duration
            if fps is not None:
                project.fps = fps
            draft.cursor = min(draft.cursor, project.duration)

        field = "duration" if duration is not None else "fps"
        self.edit(operation, merge_key=("timing", field) if duration is None or fps is None else None)

    def set_settings(self, **values: float | int) -> None:
        settings = SolverSettings.model_validate({**self._state.project.settings.model_dump(), **values})
        merge_key = (
            ("settings", next(iter(values)))
            if len(values) == 1 and not isinstance(next(iter(values.values())), bool)
            else None
        )
        self.edit(lambda project, draft: setattr(project, "settings", settings), merge_key=merge_key)

    def set_frame_task(self, name: str, task: FrameTask) -> None:
        current = self._state.project.frame_tasks.get(name, FrameTask())

        def operation(project: Project, draft: Draft) -> None:
            project.frame_tasks[name] = task.model_copy(deep=True)

        # 模式和参考系切换各自撤销；同一任务的连续权重调整合并。
        merge_key = (
            ("frame_task", name) if (current.mode, current.reference_body) == (task.mode, task.reference_body) else None
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

        self.edit(operation)
        self.finish_edit()

    def select_key(self, key_id: str | None) -> None:
        self.finish_edit()
        if key_id is None:
            if self._state.draft.selected_key is not None:
                self._state.draft.selected_key = None
                self.revision += 1
            return
        key = next(k for k in self._state.project.keyframes if k.id == key_id)
        self._state.draft.selected_key = key.id
        self._state.draft.cursor = key.time
        self._state.draft.qpos = np.asarray(key.qpos).copy()
        self._state.draft.constraints.reanchor(self.robot, self._state.draft.qpos)
        self.revision += 1

    def update_key(
        self, key_id: str, *, time: float | None = None, name: str | None = None, locked: bool | None = None
    ) -> None:
        def operation(project: Project, draft: Draft) -> None:
            key = next(k for k in project.keyframes if k.id == key_id)
            if time is not None:
                key.time = time
                if draft.selected_key == key_id:
                    draft.cursor = time
            if name is not None:
                key.name = name
            if locked is not None:
                key.locked = locked

        fields = [field for field, value in {"time": time, "name": name, "locked": locked}.items() if value is not None]
        merge_key = (fields[0], key_id) if len(fields) == 1 and fields[0] != "locked" else None
        self.edit(operation, merge_key=merge_key)

    def delete_key(self, key_id: str) -> None:
        def operation(project: Project, draft: Draft) -> None:
            project.keyframes = [key for key in project.keyframes if key.id != key_id]
            if draft.selected_key == key_id:
                draft.selected_key = None

        self.edit(operation)

    def _set_pose(self, qpos: FloatArray, *, merge: bool = True, constraints: EditConstraints | None = None) -> None:
        def operation(project: Project, draft: Draft) -> None:
            if draft.selected_key is not None:
                key = next(key for key in project.keyframes if key.id == draft.selected_key)
                if abs(key.time - draft.cursor) > 1e-8:
                    draft.selected_key = None
                else:
                    key.qpos = qpos.tolist()
            draft.qpos = qpos.copy()
            if constraints is not None:
                draft.constraints = constraints.model_copy(deep=True)

        self.edit(operation, merge_key=("pose", self._state.draft.selected_key or "draft") if merge else None)

    def set_joint_locked(self, name: str, locked: bool) -> None:
        if name not in self.robot.joint_names:
            raise ValueError(f"未知关节：{name}")

        def operation(project: Project, draft: Draft) -> None:
            joints = draft.constraints.locked_joints
            if locked and name not in joints:
                joints.append(name)
            elif not locked and name in joints:
                joints.remove(name)

        self.edit(operation)

    def unlock_all_joints(self) -> None:
        self.edit(lambda project, draft: draft.constraints.locked_joints.clear())

    def set_anchor(self, name: str, mode: ControlMode | None) -> None:
        self.robot.frame(name)

        def operation(project: Project, draft: Draft) -> None:
            if mode is None:
                draft.constraints.anchors.pop(name, None)
            elif name not in draft.constraints.anchors or draft.constraints.anchors[name].mode != mode:
                draft.constraints.capture(self.robot, self.robot.forward(draft.qpos), name, mode)

        self.edit(operation)

    def clear_fixed_parts(self) -> None:
        self.edit(lambda project, draft: draft.constraints.anchors.clear())

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
