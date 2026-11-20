"""按当前足底位置自动识别平地接触并计算准静态支撑区域。"""

from dataclasses import dataclass

import mujoco as mj
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull, QhullError

from robot_motion_forge.utils.array_types import FloatArray

from . import GROUND_HEIGHT_M
from .model import Robot

CONTACT_POSITION_TOLERANCE_M = 0.01
CONTACT_ROTATION_TOLERANCE_RAD = np.deg2rad(5.0)


@dataclass(frozen=True)
class FootState:
    name: str
    vertices_m: FloatArray
    normal: FloatArray

    @property
    def height_m(self) -> float:
        return float(np.max(np.abs(self.vertices_m[:, 2] - GROUND_HEIGHT_M)))

    @property
    def tilt_rad(self) -> float:
        return float(np.arccos(np.clip(abs(self.normal[2]), 0.0, 1.0)))


def foot_states(robot: Robot, data: mj.MjData, *, names: tuple[str, ...] | None = None) -> tuple[FootState, ...]:
    feet: list[FootState] = []
    for name, polygon in robot.support_polygons.items():
        if names is not None and name not in names:
            continue
        site = polygon.site_id
        position = data.site_xpos[site, :]
        matrix = data.site_xmat[site, :].reshape(3, 3)
        feet.append(FootState(name, position + polygon.vertices_m @ matrix.T, matrix @ polygon.normal))
    return tuple(feet)


def detect_support_contacts(feet: tuple[FootState, ...]) -> tuple[FootState, ...]:
    return tuple(
        foot
        for foot in feet
        if foot.height_m <= CONTACT_POSITION_TOLERANCE_M and foot.tilt_rad <= CONTACT_ROTATION_TOLERANCE_RAD
    )


@dataclass(frozen=True)
class SupportRegion:
    contacts: tuple[FootState, ...]
    points_m: FloatArray
    hull_indices: NDArray[np.intp]
    normals: FloatArray
    bounds: FloatArray

    @property
    def vertices_m(self) -> FloatArray:
        vertices = self.points_m[self.hull_indices].copy()
        vertices[:, 2] = GROUND_HEIGHT_M
        return vertices

    def edge_margins(self, com_xy: FloatArray) -> FloatArray:
        return self.bounds - self.normals @ com_xy


def build_support_region(contacts: tuple[FootState, ...]) -> SupportRegion | None:
    if not contacts:
        return None
    points = np.concatenate([contact.vertices_m for contact in contacts])
    try:
        hull = np.asarray(ConvexHull(points[:, :2]).vertices, dtype=int)
    except QhullError:
        return None
    vertices = points[hull, :2]
    edges = np.roll(vertices, -1, axis=0) - vertices
    normals = np.column_stack((edges[:, 1], -edges[:, 0]))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    return SupportRegion(contacts, points, hull, normals, np.sum(normals * vertices, axis=1))
