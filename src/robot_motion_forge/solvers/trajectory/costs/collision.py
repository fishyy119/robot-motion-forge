"""用 MuJoCo 距离与接触候选构造 PyCeres 碰撞残差和解析雅可比。"""

from collections import defaultdict
from itertools import combinations
from threading import Lock
from weakref import WeakKeyDictionary

import mujoco as mj
import numpy as np
import pyceres

from robot_motion_forge.robot.kinematics import forward_kinematics
from robot_motion_forge.robot.model import Robot
from robot_motion_forge.utils.array_types import FloatArray

from ..parameters import Jacobians, Parameters, PoseLayout, write_jacobians

# 固定的归一化距离尺度（1 cm），独立于自碰撞安全间隙。
COLLISION_DISTANCE_SCALE_M = 0.01
# 判断接触候选是否覆盖最深距离时允许的数值误差。
GROUND_CONTACT_DISTANCE_TOLERANCE_M = 1e-9


class CollisionModel:
    """纯机器人与固定地面的 MuJoCo 碰撞模型，各求解节点共享。"""

    def __init__(self, robot: Robot) -> None:
        self.model = robot.model_with_ground
        self.collision_pairs = np.asarray(self._collision_pairs(), dtype=np.intp).reshape(-1, 2)
        # 纯机器人不含 world 几何，与 world 的碰撞即地面碰撞。
        self.ground_collision_mask = np.any(self.model.geom_bodyid[self.collision_pairs] == 0, axis=1)
        self._plane_mask = np.any(self.model.geom_type[self.collision_pairs] == mj.mjtGeom.mjGEOM_PLANE, axis=1)
        self.ground_row_by_pair: dict[tuple[int, int], int] = {}
        for row in np.flatnonzero(self.ground_collision_mask):
            first, second = (int(geom) for geom in self.collision_pairs[row])
            if mj.mj_maxContact(self.model, first, second, -1) > 1:
                self.ground_row_by_pair[first, second] = int(row)
        self.ground_multi_contact_rows = np.fromiter(self.ground_row_by_pair.values(), dtype=np.intp)

    def _collision_pairs(self) -> list[tuple[int, int]]:
        m = self.model
        geoms = [i for i in range(m.ngeom) if m.geom_contype[i] or m.geom_conaffinity[i]]
        supported = {
            int(x)
            for x in (
                mj.mjtGeom.mjGEOM_PLANE,
                mj.mjtGeom.mjGEOM_SPHERE,
                mj.mjtGeom.mjGEOM_CAPSULE,
                mj.mjtGeom.mjGEOM_ELLIPSOID,
                mj.mjtGeom.mjGEOM_CYLINDER,
                mj.mjtGeom.mjGEOM_BOX,
                mj.mjtGeom.mjGEOM_MESH,
            )
        }
        if any(int(m.geom_type[i]) not in supported for i in geoms):
            raise ValueError("距离约束暂不支持 heightfield/SDF；请使用网格或基本碰撞几何")
        pairs: set[tuple[int, int]] = set()
        for a, b in combinations(geoms, 2):
            ba, bb = int(m.geom_bodyid[a]), int(m.geom_bodyid[b])
            wa, wb = int(m.body_weldid[ba]), int(m.body_weldid[bb])
            if wa == wb:
                continue
            if (
                not (m.opt.disableflags & mj.mjtDisableBit.mjDSBL_FILTERPARENT)
                and wa
                and wb
                and (m.body_weldid[m.body_parentid[wa]] == wb or m.body_weldid[m.body_parentid[wb]] == wa)
            ):
                continue
            signature = (min(ba, bb) << 16) + max(ba, bb)
            if signature in m.exclude_signature:
                continue
            if (m.geom_contype[a] & m.geom_conaffinity[b]) or (m.geom_contype[b] & m.geom_conaffinity[a]):
                pairs.add((a, b))
        pairs.update(
            (min(int(a), int(b)), max(int(a), int(b))) for a, b in zip(m.pair_geom1, m.pair_geom2, strict=True)
        )
        return sorted(pairs)

    def collision_errors(
        self,
        data: mj.MjData,
        margin: float,
        *,
        jacobian: FloatArray | None = None,
    ) -> FloatArray:
        """固定维度距离残差；包围球先筛选，不忽略尚未产生 contact 的临近几何。"""
        m, pairs = self.model, self.collision_pairs
        errors = np.zeros(len(pairs))
        if jacobian is not None:
            jacobian.fill(0.0)
        if len(pairs) == 0:
            return errors
        a, b = pairs.T
        plane = self._plane_mask
        bound = np.linalg.norm(data.geom_xpos[a] - data.geom_xpos[b], axis=1) - m.geom_rbound[a] - m.geom_rbound[b]
        fromto = np.empty(6) if jacobian is not None else None
        ja, jb = np.empty((3, m.nv)), np.empty((3, m.nv))
        for i in np.flatnonzero(plane | (bound < margin)):
            # 地面允许零间隙接触；其他几何保留安全间隙。
            clearance = 0.0 if plane[i] else margin
            distance = mj.mj_geomDistance(m, data, int(a[i]), int(b[i]), max(margin, 1e-6), fromto)
            errors[i] = max(0.0, clearance - distance)
            if jacobian is None or fromto is None or errors[i] == 0.0:
                continue
            normal = fromto[3:] - fromto[:3]
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                # 接触点重合时，平面仍有确定法线；其他退化接触取中心连线。
                if plane[i]:
                    first = m.geom_type[a[i]] == mj.mjtGeom.mjGEOM_PLANE
                    geom = a[i] if first else b[i]
                    normal = data.geom_xmat[geom].reshape(3, 3)[:, 2] * (1 if first else -1)
                else:
                    normal = data.geom_xpos[b[i]] - data.geom_xpos[a[i]]
                    if np.linalg.norm(normal) < 1e-12:
                        normal = np.array([1.0, 0.0, 0.0])
                norm = np.linalg.norm(normal)
            normal = normal / norm
            mj.mj_jac(m, data, ja, None, fromto[:3], int(m.geom_bodyid[a[i]]))
            mj.mj_jac(m, data, jb, None, fromto[3:], int(m.geom_bodyid[b[i]]))
            # 穿透时最近点连线反向；残差是 clearance - signed_distance。
            jacobian[i] = (-1 if distance >= 0 else 1) * (normal @ (jb - ja))
        return errors


_collision_models: WeakKeyDictionary[Robot, CollisionModel] = WeakKeyDictionary()
_collision_models_lock = Lock()


def collision_model_for(robot: Robot) -> CollisionModel:
    """首次使用时编译，随后由同一机器人的碰撞求解复用。"""
    with _collision_models_lock:
        model = _collision_models.get(robot)
        if model is None:
            model = CollisionModel(robot)
            _collision_models[robot] = model
        return model


class CollisionCost(pyceres.CostFunction):
    def __init__(self, layout: PoseLayout, collision: CollisionModel, weights: FloatArray, margin_cm: float) -> None:
        super().__init__()
        self.layout, self.collision = layout, collision
        # 单点残差 r = sqrt(w / N) * max(0, clearance - signed_distance) / distance_scale。
        # 地面多点残差在缩放前对各接触点的穿透深度取四范数。
        # 残差和解析雅可比共用缩放，N 为轨迹采样帧数。
        self.weights = weights / COLLISION_DISTANCE_SCALE_M
        # 工程约定使用 cm，MuJoCo 距离计算使用 m。
        self.margin_m = margin_cm / 100.0
        model = collision.model
        self.data = mj.MjData(model)
        self.derivative = np.empty((len(weights), model.nv))
        rows = collision.ground_multi_contact_rows
        self._ground_rows = rows[weights[rows] > 0]
        self.set_num_residuals(len(weights))
        self.set_parameter_block_sizes(self.layout.sizes)

    def Evaluate(self, parameters: Parameters, residuals: FloatArray, jacobians: Jacobians) -> bool:
        pose = self.layout.unpack(parameters)
        forward_kinematics(self.collision.model, pose, self.data)
        derivative = self.derivative if jacobians is not None else None
        errors = self.collision.collision_errors(self.data, self.margin_m, jacobian=derivative)
        if self._ground_rows.size and np.any(errors[self._ground_rows] > 0):
            self._aggregate_ground_contacts(errors, derivative)
        residuals[:] = errors * self.weights
        if derivative is not None:
            write_jacobians(jacobians, self.layout.jacobian_blocks(derivative * self.weights[:, None], parameters))
        return True

    def _aggregate_ground_contacts(self, errors: FloatArray, derivative: FloatArray | None) -> None:
        model, data = self.collision.model, self.data
        mj.mj_collision(model, data)
        contacts_by_row: dict[int, list[mj.MjContact]] = defaultdict(list)
        for contact in data.contact:
            first, second = (int(geom) for geom in contact.geom)
            row = self.collision.ground_row_by_pair.get((min(first, second), max(first, second)))
            if row is not None and self.weights[row] > 0 and errors[row] > 0 and contact.dist < 0:
                contacts_by_row[row].append(contact)

        point_jacobian = np.empty((3, model.nv)) if derivative is not None else None
        for row, contacts in contacts_by_row.items():
            if len(contacts) < 2:
                continue
            depths = np.asarray([-contact.dist for contact in contacts], dtype=float)
            # 接触筛选可能漏掉最近点；多点结果与距离查询不一致时保留单点残差。
            if abs(float(depths.max()) - errors[row]) > GROUND_CONTACT_DISTANCE_TOLERANCE_M:
                continue
            depth = float(np.linalg.norm(depths, ord=4))
            errors[row] = depth
            if derivative is None or point_jacobian is None:
                continue
            derivative[row].fill(0.0)
            # d||p||₄/dpᵢ = (pᵢ / ||p||₄)³；从接触中点还原机器人一侧的点。
            for contact, factor in zip(contacts, (depths / depth) ** 3, strict=True):
                body_a, body_b = (int(model.geom_bodyid[geom]) for geom in contact.geom)
                body = body_b if body_a == 0 else body_a
                normal = contact.frame[:3] if body_a == 0 else -contact.frame[:3]
                point = contact.pos + 0.5 * contact.dist * normal
                mj.mj_jac(model, data, point_jacobian, None, point, body)
                derivative[row] -= factor * (normal @ point_jacobian)
