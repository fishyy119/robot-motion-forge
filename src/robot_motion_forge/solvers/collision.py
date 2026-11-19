"""用 MuJoCo 距离与接触候选构造 PyCeres 碰撞残差和解析雅可比。"""

from collections import defaultdict

import mujoco as mj
import numpy as np
import pyceres

from robot_motion_forge.robot.model import FloatArray, Robot

from .trajectory_costs import Jacobians, Parameters, PoseLayout, write_jacobians

# 固定的归一化距离尺度（1 cm），独立于自碰撞安全间隙。
COLLISION_DISTANCE_SCALE_M = 0.01
# 判断接触候选是否覆盖最深距离时允许的数值误差。
GROUND_CONTACT_DISTANCE_TOLERANCE_M = 1e-9


class CollisionCost(pyceres.CostFunction):
    def __init__(self, robot: Robot, weights: FloatArray, margin_cm: float) -> None:
        super().__init__()
        self.environment = robot.environment
        # 单点残差 r = sqrt(w / N) * max(0, clearance - signed_distance) / distance_scale。
        # 地面多点残差在缩放前对各接触点的穿透深度取四范数。
        # 残差和解析雅可比共用缩放，N 为轨迹采样帧数。
        self.weights = weights / COLLISION_DISTANCE_SCALE_M
        # 工程约定使用 cm，MuJoCo 距离计算使用 m。
        self.margin_m = margin_cm / 100.0
        model = self.environment.model
        self.data = mj.MjData(model)
        self.derivative = np.empty((len(weights), robot.model.nv))
        self.layout = PoseLayout(robot)
        self._ground_row_by_pair: dict[tuple[int, int], int] = {}
        for row in np.flatnonzero(self.environment.ground_collision_mask & (weights > 0)):
            first, second = self.environment.collision_pairs[row]
            if mj.mj_maxContact(model, first, second, -1) > 1:
                self._ground_row_by_pair[first, second] = int(row)
        self._ground_rows = np.fromiter(self._ground_row_by_pair.values(), dtype=np.intp)
        self.set_num_residuals(len(weights))
        self.set_parameter_block_sizes(self.layout.sizes)

    def Evaluate(self, parameters: Parameters, residuals: FloatArray, jacobians: Jacobians) -> bool:
        pose = self.layout.unpack(parameters)
        self.environment.forward(pose, self.data)
        derivative = self.derivative if jacobians is not None else None
        errors = self.environment.collision_errors(self.data, self.margin_m, jacobian=derivative)
        if self._ground_rows.size and np.any(errors[self._ground_rows] > 0):
            self._aggregate_ground_contacts(errors, derivative)
        residuals[:] = errors * self.weights
        if derivative is not None:
            write_jacobians(jacobians, self.layout.jacobian_blocks(derivative * self.weights[:, None], parameters))
        return True

    def _aggregate_ground_contacts(self, errors: FloatArray, derivative: FloatArray | None) -> None:
        model, data = self.environment.model, self.data
        mj.mj_collision(model, data)
        contacts_by_row: dict[int, list[mj.MjContact]] = defaultdict(list)
        for contact in data.contact:
            first, second = (int(geom) for geom in contact.geom)
            row = self._ground_row_by_pair.get((min(first, second), max(first, second)))
            if row is not None and errors[row] > 0 and contact.dist < 0:
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
