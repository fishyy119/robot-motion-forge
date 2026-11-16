"""用 MuJoCo 原始碰撞几何构造 PyCeres 距离残差和解析雅可比。"""

import mujoco as mj
import numpy as np
import pyceres

from robot_motion_forge.robot.model import FloatArray, Robot

from .trajectory_costs import Jacobians, Parameters, PoseLayout, write_jacobians

# 固定的归一化距离尺度（1 cm），独立于自碰撞安全间隙。
COLLISION_DISTANCE_SCALE_M = 0.01


class CollisionCost(pyceres.CostFunction):
    def __init__(self, robot: Robot, weights: FloatArray, margin_cm: float) -> None:
        super().__init__()
        self.environment = robot.environment
        # r = sqrt(w / N) * max(0, clearance - signed_distance) / distance_scale。
        # 残差和解析雅可比共用缩放，N 为轨迹采样帧数。
        self.weights = weights / COLLISION_DISTANCE_SCALE_M
        # 工程约定使用 cm，MuJoCo 距离计算使用 m。
        self.margin_m = margin_cm / 100.0
        self.data = mj.MjData(self.environment.model)
        self.derivative = np.empty((len(weights), robot.model.nv))
        self.layout = PoseLayout(robot)
        self.set_num_residuals(len(weights))
        self.set_parameter_block_sizes(self.layout.sizes)

    def Evaluate(self, parameters: Parameters, residuals: FloatArray, jacobians: Jacobians) -> bool:
        pose = self.layout.unpack(parameters)
        self.environment.forward(pose, self.data)
        residuals[:] = (
            self.environment.collision_errors(
                self.data, self.margin_m, jacobian=self.derivative if jacobians is not None else None
            )
            * self.weights
        )
        if jacobians is not None:
            derivative = self.derivative * self.weights[:, None]
            write_jacobians(jacobians, self.layout.jacobian_blocks(derivative, parameters))
        return True
