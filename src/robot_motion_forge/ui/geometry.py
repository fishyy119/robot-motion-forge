"""将 MuJoCo 几何直接映射为 Viser 节点，仅为缺少原生接口的类型生成网格。"""

# Trimesh 创建基本几何的函数仍有未注解的 **kwargs，仅屏蔽这类未知成员诊断。
# pyright: reportUnknownMemberType=false

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco as mj
import numpy as np
import trimesh

if TYPE_CHECKING:
    from typing import TypedDict

    from viser import BoxHandle, CylinderHandle, IcosphereHandle, MeshHandle, SceneApi

    from robot_motion_forge.robot.model import FloatArray

    class _GeometryProps(TypedDict):
        color: tuple[int, int, int]
        opacity: float
        position: FloatArray
        wxyz: FloatArray
        visible: bool


type GeometryHandle = MeshHandle | BoxHandle | IcosphereHandle | CylinderHandle


def add_geom(
    scene: SceneApi,
    model: mj.MjModel,
    index: int,
    *,
    name: str,
    color: tuple[int, int, int],
    opacity: float,
    visible: bool,
) -> GeometryHandle | None:
    kind = model.geom_type[index]
    size = model.geom_size[index]
    props: _GeometryProps = {
        "color": color,
        "opacity": opacity,
        "position": model.geom_pos[index],
        "wxyz": model.geom_quat[index],
        "visible": visible,
    }
    if kind == mj.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[index]
        va, vn = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        fa, fn = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
        return scene.add_mesh_simple(
            name,
            vertices=model.mesh_vert[va : va + vn],
            faces=np.asarray(model.mesh_face[fa : fa + fn], dtype=np.uint32),
            # 原始网格按面着色，否则锐边会因为平滑产生显示问题。
            flat_shading=True,
            **props,
        )
    if kind == mj.mjtGeom.mjGEOM_BOX:
        return scene.add_box(name, dimensions=2 * size, **props)
    if kind == mj.mjtGeom.mjGEOM_SPHERE:
        return scene.add_icosphere(name, radius=float(size[0]), **props)
    if kind == mj.mjtGeom.mjGEOM_ELLIPSOID:
        return scene.add_icosphere(name, radius=1.0, scale=size, **props)
    if kind == mj.mjtGeom.mjGEOM_CYLINDER:
        return scene.add_cylinder(name, radius=float(size[0]), height=2 * float(size[1]), radial_segments=24, **props)
    if kind == mj.mjtGeom.mjGEOM_CAPSULE:
        mesh = trimesh.creation.capsule(radius=size[0], height=2 * size[1], count=[12, 12])
        mesh.vertices -= mesh.bounds.mean(axis=0)
        return scene.add_mesh_simple(
            name,
            vertices=np.asarray(mesh.vertices, dtype=np.float32),
            faces=np.asarray(mesh.faces, dtype=np.uint32),
            **props,
        )
    if kind == mj.mjtGeom.mjGEOM_PLANE:
        # 无限平面沿用有限薄盒显示，地面网格仍由 RobotView 独立管理。
        return scene.add_box(name, dimensions=(max(4.0, 2 * size[0]), max(4.0, 2 * size[1]), 0.002), **props)
    return None
