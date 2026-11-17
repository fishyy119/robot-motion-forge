"""优化参数、坐标系任务、后台求解任务的生命周期与动作导出。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import Event
from typing import TYPE_CHECKING

from robot_motion_forge.io.export import serialize_motion
from robot_motion_forge.solvers.trajectory import SolveCancelled, optimize

if TYPE_CHECKING:
    from concurrent.futures import Executor, Future

    from viser import ClientHandle

    from robot_motion_forge.application.editor import Editor
    from robot_motion_forge.motion.trajectory import Trajectory
    from robot_motion_forge.motion.types import FrameTask

    from ..notifications import Notifications, OperationNotification
    from ..tabs.solve import SolveTab


@dataclass(frozen=True)
class _SolveJob:
    future: Future[Trajectory]
    cancel: Event
    fingerprint: str
    notification: OperationNotification
    revision: int


class SolveController:
    def __init__(
        self,
        editor: Editor,
        tab: SolveTab,
        *,
        executor: Executor,
        enqueue: Callable[[Callable[[], None], ClientHandle | None], None],
        notifications: Notifications,
        pause: Callable[[], None],
        reset_target: Callable[[], None],
        on_solution: Callable[[bool], None],
    ) -> None:
        self._editor, self._tab = editor, tab
        self._executor, self._enqueue, self._notifications = executor, enqueue, notifications
        self._pause, self._reset_target = pause, reset_target
        self._on_solution = on_solution
        self._job: _SolveJob | None = None
        self._tab.set_busy(self._job is not None)

    def set_timing(self, *, duration: float | None = None, fps: float | None = None) -> None:
        self._reset_target()
        self._editor.set_timing(duration=duration, fps=fps)

    def set_setting(self, name: str, value: float | int) -> None:
        self._editor.set_settings(**{name: value})

    def set_frame_task(self, name: str, task: FrameTask) -> None:
        self._editor.set_frame_task(name, task)

    def export(self, client: ClientHandle) -> None:
        state = self._editor.snapshot()
        payload = serialize_motion(self._editor.robot, state.project, self._editor.preview())
        # Viser 下载流程自带接收完成通知。
        client.send_file_download("motion.pkl", payload, save_immediately=True)

    def start(self) -> None:
        if self._job is not None:
            raise ValueError("已有求解任务正在执行")
        self._pause()
        self._reset_target()
        self._editor.finish_edit()
        snapshot = self._editor.snapshot()
        revision = self._editor.revision
        fingerprint = snapshot.solve_fingerprint
        client = self._notifications.client
        cancel = Event()
        notification = self._notifications.show("轨迹优化中", client=client, loading=True)

        def progress(iteration: int, cost: float) -> None:
            text = f"迭代 {iteration}/{snapshot.project.settings.max_iterations} · 代价 {cost:.5g}"
            self._enqueue(
                lambda: notification.progress(text),
                client,
            )

        try:
            future = self._executor.submit(
                optimize, self._editor.robot, snapshot.project, cancel=cancel, progress=progress
            )
        except Exception as exc:
            self._tab.set_busy(False)
            notification.finish("求解失败", str(exc), color="red")
            return
        self._job = _SolveJob(future, cancel, fingerprint, notification, revision)
        self._tab.set_busy(True)

    def cancel(self) -> None:
        if self._job is not None:
            self._job.cancel.set()

    def poll(self) -> None:
        job = self._job
        self._tab.set_busy(job is not None)
        if job is None or not job.future.done():
            return
        self._job = None
        self._tab.set_busy(False)
        try:
            if job.cancel.is_set():
                raise SolveCancelled("已取消求解")
            trajectory = job.future.result()
            refresh_pose = self._editor.revision == job.revision
            if self._editor.install_solution(trajectory, job.fingerprint):
                self._on_solution(refresh_pose)
                job.notification.finish(
                    "轨迹优化完成" if trajectory.converged else "轨迹未收敛",
                    trajectory.message if trajectory.converged else f"已保留候选结果 · {trajectory.message}",
                    color="green" if trajectory.converged else "orange",
                )
            else:
                job.notification.finish("求解结果已丢弃", "求解期间工程已修改", color="orange")
        except SolveCancelled:
            job.notification.finish("求解已取消", color="orange")
        except Exception as exc:
            job.notification.finish("求解失败", str(exc), color="red")

    def close(self, reason: str = "") -> None:
        # 工程重建后不再轮询旧任务；取消并完成其通知，后台只持有旧快照。
        job, self._job = self._job, None
        self._tab.set_busy(False)
        if job is not None:
            job.cancel.set()
            job.future.cancel()
            job.notification.finish("求解已取消", reason, color="orange")
