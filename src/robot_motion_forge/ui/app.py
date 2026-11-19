"""Viser 编辑器。回调只入队，主线程更新会话，后台线程只求解不可变快照。"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue
from typing import TYPE_CHECKING, NamedTuple

from robot_motion_forge.application.editor import Editor
from robot_motion_forge.application.workspace import Workspace

from . import Controls, DragControls, Notifications
from .controllers import (
    ArrangementController,
    PlaybackController,
    PoseController,
    SolveController,
)
from .layout import PlaybackPanel, Sidebar
from .theme import apply_theme
from .viewer import RobotView
from .viser_bridge import register_browser_extensions

if TYPE_CHECKING:
    from viser import ClientHandle, IcosphereHandle, SceneClickEvent, SceneNodePointerEvent

    from .tabs.display import DisplayState


class _QueuedAction(NamedTuple):
    epoch: int
    action: Callable[[], None]
    client: ClientHandle | None


class ForgeApp:
    def __init__(
        self,
        editor: Editor,
        *,
        workspace: Workspace,
        host: str = "127.0.0.1",
        port: int = 8080,
        project_name: str | None = None,
    ) -> None:
        import viser

        self._workspace = workspace
        project_path = workspace.project_path(project_name)
        self.server = viser.ViserServer(host=host, port=port)
        register_browser_extensions(self.server)
        self.editor = editor
        # 客户端随操作入队，操作通知发给触发操作的浏览器。
        self._events: Queue[_QueuedAction] = Queue()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="trajectory-solver")
        self._notifications = Notifications(self.server)
        self.dark_mode = True
        self._epoch = 0
        # 运行期间机器人固定；场景和拖拽控件不随工程界面重建。
        self.view = RobotView(self.server, editor.robot)
        self.drag = DragControls(self.server)
        self._build(project_path)

        def on_connect(client: ClientHandle) -> None:
            client.camera.position = (1.2, -1.5, 1.0)
            client.camera.look_at = (0.0, 0.0, 0.35)
            client.camera.up_direction = (0.0, 0.0, 1.0)

        self.server.on_client_connect(on_connect)

        def on_scene_click(event: SceneClickEvent) -> None:
            # 原生坐标系捕获的手势不会进入场景点击；拾取与姿态更新统一在主线程执行。
            epoch = self._epoch
            self._events.put(
                _QueuedAction(
                    epoch=epoch,
                    action=lambda: self.pose.select_at(event.ray_origin, event.ray_direction),
                    client=event.client,
                )
            )

        self.server.scene.on_click()(on_scene_click)

        async def on_marker_click(event: SceneNodePointerEvent[IcosphereHandle]) -> None:
            epoch = self._epoch
            frame = event.target.name.removeprefix("/frames/")
            self._events.put(
                _QueuedAction(epoch=epoch, action=lambda: self.pose.select_object(frame), client=event.client)
            )

        for marker in self.view.markers.values():
            marker.on_click(on_marker_click)

    def _build(self, project_path: Path) -> None:
        self._epoch += 1
        e, gui = self.editor, self.server.gui
        self._display_revision = -1
        self._display_state: DisplayState | None = None
        epoch = self._epoch

        def enqueue(action: Callable[[], None], client: ClientHandle | None) -> None:
            self._events.put(_QueuedAction(epoch=epoch, action=action, client=client))

        controls = Controls(gui, enqueue)
        project, draft = e.project_snapshot(), e.draft_snapshot()
        self._display_project = project
        self._display_project_revision = e.project_revision
        self.sidebar = Sidebar(controls, robot=e.robot, project=project, qpos=draft.qpos, project_path=project_path)
        self.playback_panel = PlaybackPanel(controls, project=project, cursor=draft.cursor)
        self.playback = PlaybackController(e, reset_target=lambda: self.pose.reset_target())
        self.pose = PoseController(
            e,
            self.sidebar.pose,
            self.view,
            self.drag,
            pause=self.playback.pause,
            notifications=self._notifications,
        )
        self.arrangement = ArrangementController(
            e,
            pause=self.playback.pause,
            reset_target=self.pose.reset_target,
            notifications=self._notifications,
        )
        self.solve = SolveController(
            e,
            self.sidebar.solve,
            executor=self._executor,
            enqueue=enqueue,
            notifications=self._notifications,
            pause=self.playback.pause,
            reset_target=self.pose.reset_target,
            on_solution=self._solution_installed,
        )
        self.sidebar.project.bind_actions(
            e,
            workspace=self._workspace,
            replace_project=self._replace_project,
            notifications=self._notifications,
        )
        self._bind_actions()
        self._apply_theme()

    def _bind_actions(self) -> None:
        self.sidebar.arrangement.bind_actions(self.arrangement)
        self.sidebar.pose.bind_actions(self.pose)
        self.sidebar.solve.bind_actions(self.solve)
        self.playback_panel.timeline.bind_actions(
            seek=self.playback.seek,
            select_key=self.arrangement.select_key,
            set_source=self.playback.set_source,
            toggle_play=self.playback.toggle,
        )
        self.playback_panel.display.bind_actions(
            change_theme=self._change_theme,
            refresh=self._refresh,
        )

    def _replace_project(self, editor: Editor, path: Path) -> None:
        self.solve.close("工程已重新加载")
        self.drag.hide()
        self.editor = editor
        self.server.gui.reset()
        self.view.bind_gui()
        self._build(path)

    def _change_theme(self) -> None:
        self.dark_mode = not self.dark_mode
        self._apply_theme()

    def _apply_theme(self) -> None:
        with self.server.atomic():
            self.playback_panel.display.set_theme(dark_mode=self.dark_mode)
            apply_theme(self.server.gui, dark_mode=self.dark_mode)
            self.view.set_theme(dark_mode=self.dark_mode)
            self.drag.set_theme(dark_mode=self.dark_mode)
        self._refresh(force=True)

    def _refresh(self, *, force: bool = False) -> None:
        e = self.editor
        self.pose.sync_inputs()
        # 初始值、用户输入和程序赋值都读取同一份控件状态，不依赖输入回调补齐场景状态。
        display = self.playback_panel.display.state
        if (
            not force
            and self._display_revision == e.revision
            and not self.pose.dirty
            and self._display_state == display
        ):
            return
        project_changed = self._display_project_revision != e.project_revision
        if project_changed:
            self._display_project = e.project_snapshot()
        project, draft = self._display_project, e.draft_snapshot()
        with self.server.atomic():
            if force or project_changed:
                self.playback_panel.timeline.refresh(project, draft.cursor)
                self.sidebar.arrangement.refresh(project, selected_key=draft.selected_key, cursor=draft.cursor)
                self.sidebar.solve.refresh(project)
            else:
                self.sidebar.arrangement.sync_key(project, selected_key=draft.selected_key, cursor=draft.cursor)
            self.sidebar.joint_curves.refresh(e.trajectory, draft.cursor)
            self.playback_panel.timeline.sync_cursor(draft.cursor, draft.pose_source)
            self.pose.sync(draft)
            if self._display_state is None or self._display_state.collisions != display.collisions:
                self.view.show_collisions(display.collisions)
            self.view.update_support(
                show_center_of_mass=display.center_of_mass,
                show_support=display.support,
                margin_cm=project.settings.support_margin_cm,
            )
        self._display_revision = e.revision
        self._display_project_revision = e.project_revision
        self._display_state = display

    def _solution_installed(self, refresh_pose: bool) -> None:
        # 后台结果只刷新未被用户改动的草稿，避免覆盖求解期间的姿态编辑或时间定位。
        if refresh_pose and self.editor.pose_source == "trajectory" and not self.pose.has_target:
            self.editor.scrub(self.editor.draft_snapshot().cursor)
            self.pose.reset_target()
        self._refresh()

    def tick(self, elapsed: float) -> None:
        for _ in range(100):
            try:
                event = self._events.get_nowait()
            except Empty:
                break
            if event.epoch == self._epoch:
                with self._notifications.for_client(event.client):
                    restore_inputs = False
                    try:
                        event.action()
                    except Exception as exc:
                        restore_inputs = True
                        self._notifications.show("操作失败", str(exc), color="red")
                    finally:
                        if event.epoch == self._epoch:
                            self._refresh(force=restore_inputs)
        self.pose.tick()
        self.solve.poll()
        self.playback.tick(elapsed)
        self._refresh()

    def run(self) -> None:
        previous = time.monotonic()
        try:
            while True:
                now = time.monotonic()
                self.tick(now - previous)
                previous = now
                time.sleep(1 / 60)
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self) -> None:
        self.solve.close()
        self._executor.shutdown(wait=True, cancel_futures=True)
        self.server.stop()
