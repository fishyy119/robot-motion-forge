"""按当前足底位置自动识别平地接触并计算准静态支撑区域。"""

from dataclasses import dataclass

import mujoco as mj
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull, QhullError

from .environment import GROUND_HEIGHT_M
from .model import FloatArray, Robot

CONTACT_POSITION_TOLERANCE_M = 0.01
CONTACT_ROTATION_TOLERANCE_RAD = np.deg2rad(5.0)
BALANCE_TOLERANCE_M = 0.001


@dataclass(frozen=True)
class FootContact:
    name: str
    vertices_m: FloatArray
    normal: FloatArray

    @property
    def height_m(self) -> float:
        return float(np.max(np.abs(self.vertices_m[:, 2] - GROUND_HEIGHT_M)))

    @property
    def tilt_rad(self) -> float:
        return float(np.arccos(np.clip(abs(self.normal[2]), 0.0, 1.0)))


def foot_contacts(robot: Robot, data: mj.MjData, *, names: tuple[str, ...] | None = None) -> tuple[FootContact, ...]:
    contacts: list[FootContact] = []
    for name, polygon in robot.support_polygons.items():
        if names is not None and name not in names:
            continue
        site = polygon.site_id
        position = data.site_xpos[site, :]
        matrix = data.site_xmat[site, :].reshape(3, 3)
        contacts.append(FootContact(name, position + polygon.vertices_m @ matrix.T, matrix @ polygon.normal))
    return tuple(contacts)


def detect_support_contacts(contacts: tuple[FootContact, ...], *, nearest: bool = False) -> tuple[FootContact, ...]:
    grounded = tuple(
        contact
        for contact in contacts
        if contact.height_m <= CONTACT_POSITION_TOLERANCE_M and contact.tilt_rad <= CONTACT_ROTATION_TOLERANCE_RAD
    )
    if grounded or not nearest or not contacts:
        return grounded
    # 求解初值未贴地时，以最接近地面的足部为修正对象；显示仍只使用真实贴地的足部。
    closest = min(contact.height_m for contact in contacts)
    return tuple(contact for contact in contacts if contact.height_m <= closest + CONTACT_POSITION_TOLERANCE_M)


@dataclass(frozen=True)
class SupportRegion:
    contacts: tuple[FootContact, ...]
    points_m: FloatArray
    hull_indices: NDArray[np.intp]
    normals: FloatArray
    bounds: FloatArray

    @property
    def vertices_m(self) -> FloatArray:
        vertices = self.points_m[self.hull_indices].copy()
        vertices[:, 2] = GROUND_HEIGHT_M
        return vertices

    def margin(self, com_xy: FloatArray) -> float:
        return float(np.min(self.bounds - self.normals @ com_xy))


def build_support_region(contacts: tuple[FootContact, ...]) -> SupportRegion | None:
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
