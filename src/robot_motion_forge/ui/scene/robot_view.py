"""将纯机器人的 MuJoCo 几何显示在 Viser，不复制或转换本地机器人资产。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco as mj
import numpy as np
from numpy.typing import ArrayLike

from robot_motion_forge.robot import GROUND_HEIGHT_M
from robot_motion_forge.robot.model import Robot
from robot_motion_forge.robot.support import (
    build_support_region,
    detect_support_contacts,
    foot_states,
)

from ..controls import PoseSelection
from ..theme import accent_color, locked_color, robot_color, warning_color
from ..viser_bridge import json_for_script
from .geometry import add_geom

if TYPE_CHECKING:
    from viser import FrameHandle, GuiHtmlHandle, IcosphereHandle, ViserServer

    from robot_motion_forge.motion.pose_constraints import EditConstraints
    from robot_motion_forge.robot.definition import ControlMode

    from .geometry import GeometryHandle


BALANCE_TOLERANCE_M = 0.001


class RobotView:
    def __init__(self, server: ViserServer, robot: Robot) -> None:
        self.server, self.robot = server, robot
        self.data = mj.MjData(robot.model)
        self.frames: dict[int, FrameHandle] = {}
        self._geometries: dict[int, GeometryHandle] = {}
        self._visual_geometries: dict[int, GeometryHandle] = {}
        self._visual_opacities: dict[int, float] = {}
        self._collision_geometries: dict[int, GeometryHandle] = {}
        self._collision_visible = False
        self._locked_joints: frozenset[str] = frozenset()
        self._anchors: dict[str, ControlMode] = {}
        self._robot_color = robot_color(True)
        self._locked_color = locked_color(True)
        self._warning_color = warning_color(True)
        self._body_joints = {body: self._selectable_joints(body) for body in robot.body_ids}
        m = robot.model
        server.scene.set_up_direction("+z")
        # 网格略低于物理地面，避免零高度接触点与网格重叠。
        self.grid = server.scene.add_grid(
            "/grid", width=4.0, height=4.0, cell_size=0.1, section_size=1.0, position=(0, 0, GROUND_HEIGHT_M - 0.002)
        )
        for body in robot.body_ids:
            self.frames[body] = server.scene.add_frame(f"/b{body}", show_axes=False)
        body_has_visual = {
            int(m.geom_bodyid[i]) for i in range(m.ngeom) if not (m.geom_contype[i] or m.geom_conaffinity[i])
        }
        for i in range(m.ngeom):
            body = int(m.geom_bodyid[i])
            collision = bool(m.geom_contype[i] or m.geom_conaffinity[i])
            visual = not collision or body not in body_has_visual
            rgba = m.geom_rgba[i]
            material = int(m.geom_matid[i])
            if material >= 0:
                rgba = m.mat_rgba[material]
            handle = add_geom(
                server.scene,
                m,
                i,
                name=f"/b{body}/g{i}",
                color=self._robot_color if visual else self._warning_color,
                # 外观视图保留配置中的透明度，RGB 统一使用主题色。
                opacity=float(rgba[3]) if visual else 1.0,
                visible=visual,
            )
            if handle is None:
                continue
            self._geometries[i] = handle
            if collision:
                self._collision_geometries[i] = handle
            if visual:
                self._visual_geometries[i] = handle
                self._visual_opacities[i] = float(rgba[3])
        self.markers: dict[str, IcosphereHandle] = {}
        for name in robot.frames:
            self.markers[name] = server.scene.add_icosphere(
                f"/frames/{name}",
                radius=0.009,
                color=(55, 210, 184),
                opacity=1.0,
                cast_shadow=False,
                receive_shadow=False,
            )
        self._support_color = accent_color(True)
        self._com = server.scene.add_icosphere("/support/com", radius=0.008, color=self._support_color, visible=False)
        self._com_projection = server.scene.add_icosphere(
            "/support/projection", radius=0.006, color=self._support_color, visible=False
        )
        self._com_line = server.scene.add_line_segments(
            "/support/com_line",
            points=np.zeros((1, 2, 3), dtype=np.float32),
            colors=self._support_color,
            thickness=1.0,
            thickness_units="screen",
            visible=False,
        )
        self._support_boundary = server.scene.add_line_segments(
            "/support/boundary",
            points=np.zeros((1, 2, 3), dtype=np.float32),
            colors=self._support_color,
            visible=False,
            thickness=2.0,
            thickness_units="screen",
        )
        self._hover_labels = self._constraint_labels()
        self.bind_gui()
        self._update_colors()

    def _selectable_joints(self, body: int) -> tuple[str, ...]:
        """网格归属最近的上游活动关节，刚性附属连杆不成为独立操作对象。"""
        robot, model = self.robot, self.robot.model
        while body:
            joints = tuple(
                name
                for joint, name in zip(robot.joint_ids, robot.joint_names, strict=True)
                if model.jnt_bodyid[joint] == body
            )
            if joints:
                return joints
            body = int(model.body_parentid[body])
        return ()

    def bind_gui(self) -> None:
        """界面重建时重新挂载提示数据，保留静态场景节点。"""
        self._hover_metadata: GuiHtmlHandle = self.server.gui.add_html("")
        self._publish_hover_labels()

    def scene_body_joints(self) -> dict[str, tuple[str, ...]]:
        return {self.frames[body].name: joints[::-1] for body, joints in self._body_joints.items()}

    def scene_frame_paths(self) -> dict[str, str]:
        return {name: marker.name for name, marker in self.markers.items()}

    def set_control_hint(self, path: str, text: str) -> None:
        if self._hover_labels.get(path) == text:
            return
        self._hover_labels[path] = text
        self._publish_hover_labels()

    def _publish_hover_labels(self) -> None:
        # 原生 GUI 状态只保留最新名称，避免切换选择时积累可重放的 JavaScript 消息。
        labels = json_for_script(self._hover_labels, ensure_ascii=False)
        self._hover_metadata.content = f'<script type="application/json" data-rmf-scene-labels>{labels}</script>'

    def _constraint_labels(self) -> dict[str, str]:
        mode_labels = {"position": "位置", "orientation": "朝向", "pose": "位姿"}
        labels: dict[str, str] = {}
        for name, marker in self.markers.items():
            mode = self._anchors.get(name)
            labels[marker.name] = f"控制坐标系 · {name}" + (f" · 固定{mode_labels[mode]}" if mode is not None else "")
        for body, joints in self._body_joints.items():
            # 同一刚体上的多个自由度共用网格，分别列出关节状态。
            labels[self.frames[body].name] = "\n".join(
                f"关节 · {name}" + (" · 已锁定" if name in self._locked_joints else "") for name in reversed(joints)
            )
        return labels

    def _update_colors(self) -> None:
        for name, marker in self.markers.items():
            marker.color = self._warning_color if name in self._anchors else self._support_color
        for geom, handle in self._visual_geometries.items():
            joints = self._body_joints[int(self.robot.model.geom_bodyid[geom])]
            handle.color = self._locked_color if self._locked_joints.intersection(joints) else self._robot_color
        if self._collision_visible:
            for handle in self._collision_geometries.values():
                handle.color = self._warning_color

    def pick(self, ray_origin: ArrayLike, ray_direction: ArrayLike) -> PoseSelection | None:
        """先拾取坐标系标记，再拾取本体，与前端的标记优先级保持一致。"""
        m = self.robot.model
        origin = np.asarray(ray_origin, dtype=float)
        direction = np.asarray(ray_direction, dtype=float)
        nearest, frame = np.inf, None
        for name, marker in self.markers.items():
            opacity = marker.opacity
            if not marker.visible or (opacity is not None and opacity <= 0):
                continue
            distance = mj.mju_rayGeom(
                np.asarray(marker.position, dtype=float),
                np.eye(3).ravel(),
                np.asarray((marker.radius, 0.0, 0.0)),
                origin,
                direction,
                mj.mjtGeom.mjGEOM_SPHERE,
            )
            if 0 <= distance < nearest:
                nearest, frame = distance, name
        if frame is not None:
            return PoseSelection("frame", frame)
        nearest, selected = np.inf, None
        for geom, handle in self._geometries.items():
            opacity = handle.opacity
            if not handle.visible or (opacity is not None and opacity <= 0):
                continue
            if m.geom_type[geom] == mj.mjtGeom.mjGEOM_MESH:
                distance = mj.mj_rayMesh(m, self.data, geom, origin, direction)
            else:
                distance = mj.mju_rayGeom(
                    self.data.geom_xpos[geom],
                    self.data.geom_xmat[geom],
                    m.geom_size[geom],
                    origin,
                    direction,
                    m.geom_type[geom],
                )
            if 0 <= distance < nearest:
                nearest, selected = distance, geom
        if selected is None:
            return None
        joints = self._body_joints[int(m.geom_bodyid[selected])]
        # 同体串联的多个自由度默认拾取最末关节，其余关节可从操作对象中单独选择。
        return PoseSelection("joint", joints[-1]) if joints else None

    def set_theme(self, *, dark_mode: bool) -> None:
        self.grid.cell_color = (58, 70, 85) if dark_mode else (216, 224, 232)
        self.grid.section_color = (97, 113, 129) if dark_mode else (154, 170, 189)
        self._robot_color = robot_color(dark_mode)
        self._locked_color = locked_color(dark_mode)
        self._warning_color = warning_color(dark_mode)
        self._support_color = accent_color(dark_mode)
        self._com.color = self._support_color
        self._update_colors()

    def update_pose(self, qpos: ArrayLike) -> None:
        self.robot.forward(qpos, self.data)
        with self.server.atomic():
            for body, frame in self.frames.items():
                frame.position = self.data.xpos[body]
                frame.wxyz = self.data.xquat[body]
            for name, marker in self.markers.items():
                position, orientation = self.robot.frame_pose(self.data, name)
                marker.position = position
                marker.wxyz = orientation

    def update_constraints(self, constraints: EditConstraints) -> None:
        locked_joints = frozenset(constraints.locked_joints)
        anchors: dict[str, ControlMode] = {name: anchor.mode for name, anchor in constraints.anchors.items()}
        if locked_joints == self._locked_joints and anchors == self._anchors:
            return
        self._locked_joints, self._anchors = locked_joints, anchors
        self._update_colors()
        self._hover_labels.update(self._constraint_labels())
        self._publish_hover_labels()

    def show_collisions(self, visible: bool) -> None:
        self._collision_visible = visible
        with self.server.atomic():
            for geom, handle in self._geometries.items():
                handle.visible = geom in (self._collision_geometries if visible else self._visual_geometries)
                if visible and geom in self._collision_geometries:
                    handle.opacity = 1.0
                elif geom in self._visual_opacities:
                    handle.opacity = self._visual_opacities[geom]
            self._update_colors()

    def update_support(self, *, show_center_of_mass: bool, show_support: bool, margin_cm: float) -> None:
        contacts = (
            detect_support_contacts(foot_states(self.robot, self.data)) if show_center_of_mass or show_support else ()
        )
        region = build_support_region(contacts)
        com = self.robot.center_of_mass(self.data)
        projection = np.array([com[0], com[1], GROUND_HEIGHT_M + 0.002])
        minimum_margin = margin_cm / 100.0 - BALANCE_TOLERANCE_M
        valid = region is not None and float(region.edge_margins(com[:2]).min()) >= minimum_margin
        color = self._support_color if valid else self._warning_color
        with self.server.atomic():
            self._com.position = com
            self._com_projection.position = projection
            self._com_projection.color = color
            self._com_line.points = np.asarray([[com, projection]], dtype=np.float32)
            self._com_line.colors = np.asarray([[color, color]], dtype=np.uint8)
            self._com.visible = self._com_projection.visible = self._com_line.visible = show_center_of_mass
            self._support_boundary.visible = show_support and region is not None
            if region is not None:
                points = region.vertices_m
                points[:, 2] += 0.002
                self._support_boundary.points = np.asarray(
                    np.stack((points, np.roll(points, -1, axis=0)), axis=1), dtype=np.float32
                )
                self._support_boundary.colors = np.tile(np.asarray(color, dtype=np.uint8), (len(points), 2, 1))
