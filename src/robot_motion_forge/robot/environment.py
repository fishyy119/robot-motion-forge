"""将纯机器人与项目固定平地组合，供轨迹优化和几何检查共用。"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, Literal

import mujoco as mj
import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from .model import FloatArray

GROUND_HEIGHT_M = 0.0


class Environment:
    def __init__(self, robot_spec: mj.MjSpec) -> None:
        spec = robot_spec.copy()
        # 地面使用独立默认值，避免继承机器人 MJCF 中的几何、材质或接触设置。
        ground = mj.MjSpec()
        ground.worldbody.add_geom(
            name="floor",
            type=mj.mjtGeom.mjGEOM_PLANE,
            pos=(0.0, 0.0, GROUND_HEIGHT_M),
            size=(0.0, 0.0, 0.05),
            friction=(1.0, 0.005, 0.0001),
            # 地面接触所有启用碰撞的机器人几何，不限定机器人使用的掩码位。
            contype=-1,
            conaffinity=-1,
        )
        # 只在 world 上添加静态几何，机器人 body、joint、actuator 和状态布局保持一致。
        spec.attach(ground, prefix="rmf_ground/", frame=spec.worldbody.add_frame())
        self.model: mj.MjModel = spec.compile()
        self.collision_pairs = self._collision_pairs()
        self._pair_array = np.asarray(self.collision_pairs, dtype=int).reshape(-1, 2)
        # 纯机器人不含 world 几何，与 world 的碰撞即地面碰撞。
        ground_pairs = np.any(self.model.geom_bodyid[self._pair_array] == 0, axis=1)
        self.self_collision_pairs = self._pair_array[~ground_pairs]
        self.ground_collision_pairs = self._pair_array[ground_pairs]

    def forward(self, qpos: ArrayLike, data: mj.MjData | None = None) -> mj.MjData:
        state = mj.MjData(self.model) if data is None else data
        state.qpos[:] = qpos
        mj.mj_kinematics(self.model, state)
        mj.mj_comPos(self.model, state)
        return state

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
        margin: float = 0.003,
        *,
        kind: Literal["self", "ground"] | None = None,
        jacobian: FloatArray | None = None,
    ) -> FloatArray:
        """固定维度距离残差；包围球先筛选，不忽略尚未产生 contact 的临近几何。"""
        m, pairs = self.model, self._pair_array
        if kind == "self":
            pairs = self.self_collision_pairs
        elif kind == "ground":
            pairs = self.ground_collision_pairs
        errors = np.zeros(len(pairs))
        if jacobian is not None:
            jacobian.fill(0.0)
        if len(pairs) == 0:
            return errors
        a, b = pairs.T
        plane = (m.geom_type[a] == mj.mjtGeom.mjGEOM_PLANE) | (m.geom_type[b] == mj.mjtGeom.mjGEOM_PLANE)
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
