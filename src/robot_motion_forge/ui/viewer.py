"""将纯机器人的 MuJoCo 几何显示在 Viser，不复制或转换本地机器人资产。"""

# Trimesh 创建基本几何的函数仍有未注解的 **kwargs，仅屏蔽这类未知成员诊断。
# pyright: reportUnknownMemberType=false

from __future__ import annotations

import json
from collections.abc import Collection
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco as mj
import numpy as np
import trimesh
from numpy.typing import ArrayLike
from scipy.sparse import coo_array
from scipy.sparse.csgraph import connected_components
from viser import _messages

from robot_motion_forge.motion.support import (
    BALANCE_TOLERANCE_M,
    build_support_region,
    detect_support_contacts,
    foot_contacts,
)
from robot_motion_forge.robot.environment import GROUND_HEIGHT_M
from robot_motion_forge.robot.model import Robot

from .theme import accent_color, locked_color, robot_color, warning_color

if TYPE_CHECKING:
    from viser import FrameHandle, GuiHtmlHandle, IcosphereHandle, MeshHandle, ViserServer


def _split_sharp_edges(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """保留 Trimesh 默认的平滑区域划分，批量重建顶点索引。"""
    adjacency = mesh.face_adjacency
    if len(adjacency) == 0:
        return mesh
    adjacency = adjacency[mesh.face_adjacency_angles < np.radians(20)]
    areas = mesh.area_faces
    min_area = mesh.area / 10
    facets = [facet for facet in mesh.facets if areas[facet].sum() > min_area]
    if facets:
        # 大平面单独保留，避免平面与周围倒角通过其他路径连成同一平滑区域。
        keep = np.ones(len(mesh.faces), dtype=bool)
        keep[np.concatenate(facets)] = False
        adjacency = adjacency[keep[adjacency].all(axis=1)]
    graph = coo_array(
        (np.ones(len(adjacency), dtype=bool), (adjacency[:, 0], adjacency[:, 1])),
        shape=(len(mesh.faces), len(mesh.faces)),
    )
    _, labels = connected_components(graph, directed=False)
    offset = int(labels.max()) + 1
    for i, facet in enumerate(facets):
        labels[facet] = offset + i
    # 相同区域内共享原顶点，跨区域复制；统一索引，避免逐区域构造子网格。
    vertex_keys = labels[:, None].astype(np.int64) * len(mesh.vertices) + mesh.faces
    unique, indices = np.unique(vertex_keys, return_inverse=True)
    return trimesh.Trimesh(mesh.vertices[unique % len(mesh.vertices)], indices.reshape((-1, 3)), process=False)


def geom_mesh(model: mj.MjModel, index: int) -> trimesh.Trimesh | None:
    kind = model.geom_type[index]
    size = model.geom_size[index]
    if kind == mj.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[index]
        va, vn = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        fa, fn = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
        mesh = trimesh.Trimesh(
            model.mesh_vert[va : va + vn].copy(),
            model.mesh_face[fa : fa + fn].copy(),
            process=False,
        )
        # Viser 会按共享顶点重算法线；断开锐边，避免跨壳体平面与孔壁平均法线。
        return _split_sharp_edges(mesh)
    if kind == mj.mjtGeom.mjGEOM_BOX:
        return trimesh.creation.box(extents=2 * size)
    if kind == mj.mjtGeom.mjGEOM_SPHERE:
        return trimesh.creation.icosphere(subdivisions=2, radius=size[0])
    if kind == mj.mjtGeom.mjGEOM_ELLIPSOID:
        mesh = trimesh.creation.icosphere(subdivisions=2)
        mesh.vertices *= size
        return mesh
    if kind == mj.mjtGeom.mjGEOM_CYLINDER:
        return trimesh.creation.cylinder(radius=size[0], height=2 * size[1], sections=24)
    if kind == mj.mjtGeom.mjGEOM_CAPSULE:
        mesh = trimesh.creation.capsule(radius=size[0], height=2 * size[1], count=[12, 12])
        mesh.vertices -= mesh.bounds.mean(axis=0)
        return mesh
    if kind == mj.mjtGeom.mjGEOM_PLANE:
        return trimesh.creation.box(extents=[max(4.0, 2 * size[0]), max(4.0, 2 * size[1]), 0.002])
    return None


class RobotView:
    def __init__(self, server: ViserServer, robot: Robot) -> None:
        self.server, self.robot = server, robot
        self.data = mj.MjData(robot.model)
        self.frames: dict[int, FrameHandle] = {}
        self._geometries: dict[int, MeshHandle] = {}
        self._visual_geometries: dict[int, MeshHandle] = {}
        self._visual_opacities: dict[int, float] = {}
        self._collision_geometries: dict[int, MeshHandle] = {}
        self._collision_visible = False
        self._locked_joints: frozenset[str] = frozenset()
        self._robot_color = robot_color(True)
        self._locked_color = locked_color(True)
        self._collision_color = warning_color(True)
        self._body_joints = {body: robot.selectable_joints(body) for body in robot.body_ids}
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
            mesh = geom_mesh(m, i)
            if mesh is None:
                continue
            body = int(m.geom_bodyid[i])
            collision = bool(m.geom_contype[i] or m.geom_conaffinity[i])
            visual = not collision or body not in body_has_visual
            rgba = m.geom_rgba[i]
            material = int(m.geom_matid[i])
            if material >= 0:
                rgba = m.mat_rgba[material]
            handle = server.scene.add_mesh_simple(
                f"/b{body}/g{i}",
                vertices=np.asarray(mesh.vertices, dtype=np.float32),
                faces=np.asarray(mesh.faces, dtype=np.uint32),
                color=self._robot_color if visual else self._collision_color,
                # 外观视图保留配置中的透明度，RGB 统一使用主题色。
                opacity=float(rgba[3]) if visual else 1.0,
                wireframe=False,
                position=m.geom_pos[i],
                wxyz=m.geom_quat[i],
                visible=visual,
            )
            self._geometries[i] = handle
            if collision:
                self._collision_geometries[i] = handle
            if visual:
                self._visual_geometries[i] = handle
                self._visual_opacities[i] = float(rgba[3])
        self.markers: dict[str, IcosphereHandle] = {}
        for name in robot.end_effectors:
            self.markers[name] = server.scene.add_icosphere(
                f"/end_effectors/{name}",
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
        self._hover_labels: dict[str, str] = {
            f"/end_effectors/{name}": f"末端坐标系 · {name}" for name in robot.end_effectors
        }
        self._hover_metadata: GuiHtmlHandle = server.gui.add_html("")
        self._update_joint_colors()
        self.set_control_hint("")

    def set_control_hint(self, text: str) -> None:
        if self._hover_labels.get("/target") == text:
            return
        self._hover_labels["/target"] = text
        self._publish_hover_labels()

    def _publish_hover_labels(self) -> None:
        # 原生 GUI 状态只保留最新名称，避免切换选择时积累可重放的 JavaScript 消息。
        labels = json.dumps(self._hover_labels, ensure_ascii=False).replace("<", "\\u003c")
        self._hover_metadata.content = f'<script type="application/json" data-rmf-scene-labels>{labels}</script>'

    def _update_joint_colors(self) -> None:
        for geom, handle in self._visual_geometries.items():
            joints = self._body_joints[int(self.robot.model.geom_bodyid[geom])]
            handle.color = self._locked_color if self._locked_joints.intersection(joints) else self._robot_color
        if self._collision_visible:
            for handle in self._collision_geometries.values():
                handle.color = self._collision_color
        for body, joints in self._body_joints.items():
            # 同一刚体上的多个自由度共用网格，分别列出关节状态。
            self._hover_labels[f"/b{body}"] = "\n".join(
                f"关节 · {name}" + (" · 已锁定" if name in self._locked_joints else "") for name in reversed(joints)
            )
        self._publish_hover_labels()

    def pick(self, ray_origin: ArrayLike, ray_direction: ArrayLike) -> str | None:
        """先拾取末端标记，再拾取本体，与前端的标记优先级保持一致。"""
        m = self.robot.model
        origin = np.asarray(ray_origin, dtype=float)
        direction = np.asarray(ray_direction, dtype=float)
        nearest, end_effector = np.inf, None
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
                nearest, end_effector = distance, name
        if end_effector is not None:
            return end_effector
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
        return f"joint:{joints[-1]}" if joints else None

    def set_theme(self, *, dark_mode: bool) -> None:
        self.grid.cell_color = (58, 70, 85) if dark_mode else (216, 224, 232)
        self.grid.section_color = (97, 113, 129) if dark_mode else (154, 170, 189)
        for marker in self.markers.values():
            marker.color = accent_color(dark_mode)
        self._robot_color = robot_color(dark_mode)
        self._locked_color = locked_color(dark_mode)
        self._collision_color = warning_color(dark_mode)
        self._support_color = accent_color(dark_mode)
        self._com.color = self._support_color
        self._update_joint_colors()

    def update(self, qpos: ArrayLike, locked_joints: Collection[str]) -> None:
        self.robot.forward(qpos, self.data)
        with self.server.atomic():
            for body, frame in self.frames.items():
                frame.position = self.data.xpos[body]
                frame.wxyz = self.data.xquat[body]
            for name, marker in self.markers.items():
                position, orientation = self.robot.pose(self.data, name)
                marker.position = position
                marker.wxyz = orientation
            current = frozenset(locked_joints)
            if current != self._locked_joints:
                self._locked_joints = current
                self._update_joint_colors()

    def show_collisions(self, visible: bool) -> None:
        self._collision_visible = visible
        with self.server.atomic():
            for geom, handle in self._geometries.items():
                handle.visible = geom in (self._collision_geometries if visible else self._visual_geometries)
                if visible and geom in self._collision_geometries:
                    handle.opacity = 1.0
                elif geom in self._visual_opacities:
                    handle.opacity = self._visual_opacities[geom]
            self._update_joint_colors()

    def update_support(self, *, show_center_of_mass: bool, show_support: bool, margin_cm: float) -> None:
        contacts = (
            detect_support_contacts(foot_contacts(self.robot, self.data)) if show_center_of_mass or show_support else ()
        )
        region = build_support_region(contacts)
        com = self.robot.center_of_mass(self.data)
        projection = np.array([com[0], com[1], GROUND_HEIGHT_M + 0.002])
        valid = region is not None and region.margin(com[:2]) >= margin_cm / 100.0 - BALANCE_TOLERANCE_M
        color = self._support_color if valid else self._collision_color
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


def configure_scene(server: ViserServer) -> None:
    # Viser 未公开小球置顶、拾取优先级和场景悬停提示，仅补足这些配置；位姿和事件仍使用原生 API。
    source = Path(__file__).with_name("scene.js").read_text(encoding="utf-8")
    server._websock_server.queue_message(_messages.RunJavascriptMessage(source=source))  # pyright: ignore[reportPrivateUsage]
