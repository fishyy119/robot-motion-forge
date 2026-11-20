"""Viser 编辑器。回调只入队，主线程更新会话，后台线程只求解不可变快照。"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from robot_motion_forge.application.editor import Editor
from robot_motion_forge.application.workspace import Workspace

from .controllers.pose import PoseController
from .controllers.solve import SolveController
from .controllers.timeline import TimelineController
from .controls import Controls
from .layout import BottomPanel, Sidebar
from .notifications import Notifications
from .scene.gizmo import DragControls
from .scene.robot_view import RobotView
from .theme import THEME_STYLE, apply_theme
from .viser_bridge import register_browser_extensions

if TYPE_CHECKING:
    from viser import ClientHandle, SceneClickEvent

    from robot_motion_forge.motion.draft import Draft
    from robot_motion_forge.motion.trajectory import SolvedTrajectory

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
        project_path: Path,
        host: str = "127.0.0.1",
        port: int = 8080,
    ) -> None:
        import viser

        self._workspace = workspace
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

    def _build(self, project_path: Path) -> None:
        self._epoch += 1
        e, gui = self.editor, self.server.gui
        self._display_revision = -1
        self._display_state: DisplayState | None = None
        self._display_draft: Draft | None = None
        self._display_trajectory: SolvedTrajectory | None = None
        epoch = self._epoch

        def enqueue(action: Callable[[], None], client: ClientHandle | None) -> None:
            self._events.put(_QueuedAction(epoch=epoch, action=action, client=client))

        controls = Controls(gui, enqueue)
        project, draft = e.project_snapshot(), e.draft_snapshot()
        self._draft, self._draft_revision = draft, e.draft_revision
        self._display_project = project
        self._display_project_revision = e.project_revision
        gui.add_html(THEME_STYLE)
        self.sidebar = Sidebar(controls, robot=e.robot, project=project, qpos=draft.qpos, project_path=project_path)
        gui.add_html(
            self.sidebar.pose.scene_menu_metadata(
                self.view.scene_frame_paths(), self.view.scene_body_joints(), gizmo_path=self.drag.gizmo.name
            )
        )
        self.bottom_panel = BottomPanel(controls, project=project, cursor=draft.cursor)
        self.timeline = TimelineController(
            e, reset_target=lambda: self.pose.reset_target(), notifications=self._notifications
        )
        self.pose = PoseController(
            e,
            self.sidebar.pose,
            self.view,
            self.drag,
            pause=self.timeline.pause,
            notifications=self._notifications,
        )
        self.solve = SolveController(
            e,
            self.sidebar.solve,
            executor=self._executor,
            enqueue=enqueue,
            notifications=self._notifications,
            pause=self.timeline.pause,
            reset_target=self.pose.reset_target,
            on_solution=self._solution_installed,
        )
        self._bind_actions()
        self._apply_theme()

    def _bind_actions(self) -> None:
        editor = self.editor
        self.sidebar.arrangement.bind_actions(self.timeline)
        self.sidebar.pose.bind_actions(self.pose)
        self.sidebar.solve.bind_actions(
            self.solve,
            set_setting=lambda name, value: editor.set_settings(**{name: value}),
            set_frame_task=editor.set_frame_task,
        )
        self.sidebar.project.bind_actions(save=self._save_project, reload=self._reload_project)
        self.bottom_panel.timeline.bind_actions(
            seek=self.timeline.seek,
            select_key=self.timeline.select_key,
            set_source=self.timeline.set_source,
            toggle_play=self.timeline.toggle,
        )
        self.bottom_panel.display.bind_actions(change_theme=self._change_theme)

    def _save_project(self, filename: str) -> str:
        path = self._workspace.project_path(filename)
        self.editor.save(path)
        self._notifications.show("工程已保存", path.name)
        return path.name

    def _reload_project(self, filename: str) -> None:
        path = self._workspace.project_path(filename)
        editor = Editor.load(self.editor.robot, path)
        self.solve.close("工程已重新加载")
        self.drag.hide()
        self.editor = editor
        self.server.gui.reset()
        self.view.bind_gui()
        self._build(path)
        self._notifications.show("工程已重新加载", path.name)

    def _change_theme(self) -> None:
        self.dark_mode = not self.dark_mode
        self._apply_theme()

    def _apply_theme(self) -> None:
        with self.server.atomic():
            self.bottom_panel.display.set_theme(dark_mode=self.dark_mode)
            apply_theme(self.server.gui, dark_mode=self.dark_mode)
            self.view.set_theme(dark_mode=self.dark_mode)
            self.drag.set_theme(dark_mode=self.dark_mode)
        self._refresh(force=True)

    def _draft_snapshot(self) -> Draft:
        revision = self.editor.draft_revision
        if revision != self._draft_revision:
            self._draft = self.editor.draft_snapshot()
            self._draft_revision = revision
        return self._draft

    def _refresh(self, *, force: bool = False) -> None:
        e = self.editor
        draft = self._draft_snapshot()
        pose_state = self.pose.sync_inputs(draft)
        # 初始值、用户输入和程序赋值都读取同一份控件状态，不依赖输入回调补齐场景状态。
        display = self.bottom_panel.display.state
        if (
            not force
            and self._display_revision == e.revision
            and not self.pose.dirty
            and self._display_state == display
        ):
            return
        project_changed = self._display_project_revision != e.project_revision
        previous_margin = self._display_project.settings.support_margin_cm
        project = e.project_snapshot() if project_changed else self._display_project
        previous = self._display_draft
        display_before = self._display_state
        qpos_changed = previous is None or (previous is not draft and not np.array_equal(previous.qpos, draft.qpos))
        constraints_changed = previous is None or (previous is not draft and previous.constraints != draft.constraints)
        cursor_changed = previous is None or previous.cursor != draft.cursor
        selection_changed = previous is None or previous.selected_key != draft.selected_key
        source_changed = previous is None or previous.pose_source != draft.pose_source
        support_changed = display_before is None or (
            (display_before.center_of_mass, display_before.support) != (display.center_of_mass, display.support)
        )
        margin = project.settings.support_margin_cm
        trajectory = e.trajectory
        with self.server.atomic():
            if force or project_changed:
                self.bottom_panel.timeline.refresh(project, draft.cursor)
                self.sidebar.arrangement.refresh(project, selected_key=draft.selected_key, cursor=draft.cursor)
                self.sidebar.solve.refresh(project)
            elif cursor_changed or selection_changed:
                self.sidebar.arrangement.sync_key(project, selected_key=draft.selected_key, cursor=draft.cursor)
            if force or cursor_changed or trajectory is not self._display_trajectory:
                self.sidebar.joint_curves.refresh(trajectory, draft.cursor)
            if force or cursor_changed or source_changed:
                self.bottom_panel.timeline.sync_cursor(draft.cursor, draft.pose_source)
            if qpos_changed:
                self.view.update_pose(draft.qpos)
            if constraints_changed:
                self.view.update_constraints(draft.constraints)
            if force or qpos_changed or constraints_changed or self.pose.dirty:
                self.pose.sync(draft, pose_state)
            if display_before is None or display_before.collisions != display.collisions:
                self.view.show_collisions(display.collisions)
            if force or qpos_changed or support_changed or margin != previous_margin:
                self.view.update_support(
                    show_center_of_mass=display.center_of_mass,
                    show_support=display.support,
                    margin_cm=margin,
                )
        self._display_revision = e.revision
        self._display_project = project
        self._display_project_revision = e.project_revision
        self._display_state = display
        self._display_draft = draft
        self._display_trajectory = trajectory

    def _solution_installed(self, refresh_pose: bool) -> None:
        # 后台结果只刷新未被用户改动的草稿，避免覆盖求解期间的姿态编辑或时间定位。
        if refresh_pose and self.editor.pose_source == "trajectory" and not self.pose.has_target:
            self.editor.scrub(self._draft_snapshot().cursor)
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
        self.pose.tick(self._draft_snapshot())
        self.solve.poll()
        self.timeline.tick(elapsed)
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
