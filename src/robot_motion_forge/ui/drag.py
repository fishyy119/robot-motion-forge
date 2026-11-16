"""拖拽输入合并与期望 / 实际位姿显示；模型更新仍由编辑器主线程执行。"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from numpy.typing import ArrayLike

from robot_motion_forge.motion.types import PoseTarget
from robot_motion_forge.robot.model import FloatArray, rotation

from .theme import accent_color, warning_color

if TYPE_CHECKING:
    from viser import ClientHandle, TransformControlsEvent, ViserServer


@dataclass(frozen=True)
class JointTarget:
    name: str
    value: float


class DragInput(NamedTuple):
    target: PoseTarget | JointTarget | None
    gesture: int
    client: ClientHandle | None


@dataclass(frozen=True)
class _JointFrame:
    name: str
    value: float
    position: FloatArray
    quaternion: FloatArray
    slide: bool


class DragControls:
    def __init__(self, server: ViserServer) -> None:
        self._lock = Lock()
        self._pending: PoseTarget | JointTarget | None = None
        self._client: ClientHandle | None = None
        self._selection: PoseTarget | _JointFrame | None = None
        self._active: PoseTarget | _JointFrame | None = None
        self._angle = self._last_angle = 0.0
        self._gesture = 0
        # 固定屏幕像素尺寸并加粗弧线，避免缩小视图后控件难以辨认和拾取。
        self.gizmo = server.scene.add_transform_controls(
            "/target",
            scale=110.0,
            fixed=True,
            line_width=4.0,
            disable_rotations=True,
            depth_test=False,
        )
        self.requested = server.scene.add_icosphere(
            "/drag/requested", radius=0.005, color=(255, 166, 64), visible=False
        )
        self.actual = server.scene.add_frame(
            "/drag/actual",
            axes_length=0.035,
            axes_radius=0.0015,
            origin_color=(55, 210, 184),
            visible=False,
        )
        self.error_line = server.scene.add_line_segments(
            "/drag/error",
            points=np.zeros((1, 2, 3), dtype=np.float32),
            colors=(255, 166, 64),
            visible=False,
        )

        async def receive(event: TransformControlsEvent) -> None:
            # 断开连接时 Viser 会补发无 client 的结束事件，仍需结束已开始的拖拽。
            if event.client is None and event.phase != "end":
                return
            # 立即保存事件位姿，避免稍后读取已被主线程更新的 handle；只保留最新请求。
            position = np.asarray(event.target.position, dtype=float).copy()
            quat = np.asarray(event.target.wxyz, dtype=float).copy()
            with self._lock:
                if event.client is not None:
                    self._client = event.client
                if event.phase == "start":
                    self._gesture += 1
                    self._active = self._selection
                    self._angle = self._last_angle = 0.0
                    return
                active = self._active
                if active is None:
                    return
                if isinstance(active, _JointFrame):
                    frame = rotation(active.quaternion)
                    if active.slide:
                        delta = float(np.dot(position - active.position, frame.apply((0.0, 0.0, 1.0))))
                    else:
                        relative = (frame.inv() * rotation(quat)).as_matrix()
                        angle = float(np.arctan2(relative[1, 0], relative[0, 0]))
                        # 累加相邻事件的最短角差，转环跨越 ±π 时关节值仍连续。
                        self._angle += (angle - self._last_angle + np.pi) % (2 * np.pi) - np.pi
                        self._last_angle = angle
                        delta = self._angle
                    self._pending = JointTarget(active.name, active.value + delta)
                else:
                    self._pending = PoseTarget(
                        active.end_effector,
                        position.tolist(),
                        quat.tolist() if active.quaternion is not None else None,
                        track_position=active.track_position,
                    )
                if event.phase == "end":
                    self._active = None

        self.gizmo.on_update(receive)

    def set_theme(self, *, dark_mode: bool) -> None:
        warning = warning_color(dark_mode)
        self.requested.color = warning
        self.actual.origin_color = accent_color(dark_mode)
        self.error_line.colors = np.asarray([[warning, warning]], dtype=np.uint8)

    def consume(self) -> DragInput:
        with self._lock:
            target, self._pending = self._pending, None
            return DragInput(target=target, gesture=self._gesture, client=self._client)

    def clear(self) -> None:
        with self._lock:
            self._pending = None
            self._active = None
            self._client = None

    def hide(self) -> None:
        self.clear()
        with self._lock:
            self._selection = None
        self.gizmo.visible = self.requested.visible = self.actual.visible = self.error_line.visible = False

    def show_joint(self, name: str, value: float, position: FloatArray, quaternion: FloatArray, *, slide: bool) -> None:
        with self._lock:
            self._selection = _JointFrame(name, value, position.copy(), quaternion.copy(), slide)
            dragging = self._active is not None
        self.gizmo.visible = True
        # 原生转环由平面内的两根活动轴启用；XY 平面对应绕 Z 轴旋转。
        self.gizmo.active_axes = (False, False, True) if slide else (True, True, False)
        self.gizmo.disable_axes = not slide
        self.gizmo.disable_sliders = True
        self.gizmo.disable_rotations = slide
        self.requested.visible = self.actual.visible = self.error_line.visible = False
        if not dragging:
            self.gizmo.position = position
            self.gizmo.wxyz = quaternion

    def show(
        self,
        end_effector: str,
        position: ArrayLike,
        quaternion: ArrayLike,
        *,
        track_position: bool = True,
        track_orientation: bool = False,
        target: PoseTarget | None = None,
    ) -> None:
        with self._lock:
            self._selection = PoseTarget(
                end_effector,
                np.asarray(position, dtype=float).tolist(),
                np.asarray(quaternion, dtype=float).tolist() if track_orientation else None,
                track_position=track_position,
            )
            dragging = self._active is not None
        self.gizmo.visible = True
        self.gizmo.active_axes = (True, True, True)
        self.gizmo.disable_axes = self.gizmo.disable_sliders = not track_position
        self.gizmo.disable_rotations = not track_orientation
        if not track_orientation:
            # 纯位置拖拽固定使用世界轴，不随末端连杆的朝向变化。
            self.gizmo.wxyz = (1.0, 0.0, 0.0, 0.0)
        visible = target is not None
        self.requested.visible = self.error_line.visible = visible and track_position
        self.actual.visible = visible
        if target is None and not dragging:
            self.gizmo.position = np.asarray(position, dtype=float)
            if track_orientation:
                self.gizmo.wxyz = np.asarray(quaternion, dtype=float)
        elif target is not None:
            # 跟随期间不把坐标轴吸回实际位置，保留鼠标期望，避免与前端拖拽相互争夺。
            if track_position:
                self.requested.position = np.asarray(target.position, dtype=float)
                self.error_line.points = np.asarray([[position, target.position]], dtype=np.float32)
            elif not dragging:
                # 纯朝向任务松手后跟随连杆原点，保留目标方向而不引入位置约束。
                self.gizmo.position = np.asarray(position, dtype=float)
                if target.quaternion is not None:
                    self.gizmo.wxyz = np.asarray(target.quaternion, dtype=float)
            self.actual.position = np.asarray(position, dtype=float)
            self.actual.wxyz = np.asarray(quaternion, dtype=float)
