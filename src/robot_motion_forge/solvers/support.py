"""足底贴地、水平与随足部运动的重心支撑区域残差。"""

import mujoco as mj
import numpy as np
import pyceres

from robot_motion_forge.motion.support import FootContact, build_support_region, foot_contacts
from robot_motion_forge.motion.types import SolverSettings
from robot_motion_forge.robot.environment import GROUND_HEIGHT_M
from robot_motion_forge.robot.model import FloatArray, Robot

from .trajectory_costs import Jacobians, Parameters, PoseLayout, write_jacobians


class SupportCost(pyceres.CostFunction):
    def __init__(self, robot: Robot, contacts: tuple[FootContact, ...], settings: SolverSettings, count: int) -> None:
        super().__init__()
        self.robot = robot
        # 每个求解节点从参考姿态识别接触脚，迭代中不因暂时离地而撤销支撑代价。
        self.names = tuple(contact.name for contact in contacts)
        # 平移与裕量按 1 cm、倾角按 1° 归一化，再按总采样帧数平均。
        self.position_scale = float(np.sqrt(settings.support_position / count)) / 0.01
        self.orientation_scale = float(np.sqrt(settings.support_orientation / count)) / np.deg2rad(1.0)
        self.balance_scale = float(np.sqrt(settings.balance / count)) / 0.01
        self.margin_m = settings.support_margin_cm / 100.0
        self.data = mj.MjData(robot.model)
        # 重心只使用最小支撑裕量，凸包边数变化时残差维度与代价保持连续。
        self.size = 3 * len(contacts) + 1
        self.layout = PoseLayout(robot)
        self.set_num_residuals(self.size)
        self.set_parameter_block_sizes(self.layout.sizes)

    def Evaluate(self, parameters: Parameters, residuals: FloatArray, jacobians: Jacobians) -> bool:
        robot = self.robot
        pose = self.layout.unpack(parameters)
        robot.forward(pose, self.data)
        contacts = foot_contacts(robot, self.data, names=self.names)
        region = build_support_region(contacts)
        residuals[:] = 0.0
        derivative = np.zeros((self.size, robot.model.nv)) if jacobians is not None else None
        vertex_jacobians: list[FloatArray] = []
        for index, contact in enumerate(contacts):
            row = 3 * index
            residuals[row] = self.position_scale * (np.mean(contact.vertices_m[:, 2]) - GROUND_HEIGHT_M)
            residuals[row + 1 : row + 3] = self.orientation_scale * contact.normal[:2]
            if derivative is not None:
                jp, jr = np.empty((3, robot.model.nv)), np.empty((3, robot.model.nv))
                site = mj.mj_name2id(robot.model, mj.mjtObj.mjOBJ_SITE, robot.end_effector_sites[contact.name])
                mj.mj_jacSite(robot.model, self.data, jp, jr, site)
                position, _ = robot.pose(self.data, contact.name)
                vertices = np.stack([jp + np.cross(jr.T, vertex - position).T for vertex in contact.vertices_m])
                vertex_jacobians.extend(vertices)
                derivative[row] = self.position_scale * np.mean(vertices[:, 2], axis=0)
                derivative[row + 1 : row + 3] = self.orientation_scale * np.cross(jr.T, contact.normal).T[:2]
        if region is not None:
            com = robot.center_of_mass(self.data)[:2]
            violations = region.normals @ com - region.bounds + self.margin_m
            index = int(np.argmax(violations))
            violation = float(violations[index])
            if violation > 0:
                row = 3 * len(contacts)
                residuals[row] = self.balance_scale * violation
            if violation > 0 and derivative is not None:
                com_jacobian = robot.center_of_mass_jacobian(self.data)[:2]
                start = int(region.hull_indices[index])
                end = int(region.hull_indices[(index + 1) % len(region.hull_indices)])
                normal = region.normals[index]
                offset = com - region.points_m[start, :2]
                first, second = vertex_jacobians[start][:2], vertex_jacobians[end][:2]
                edge = region.points_m[end, :2] - region.points_m[start, :2]
                length = float(np.linalg.norm(edge))
                # d[n·(c-a)] = n·(dc-da) + (c-a)·dn；足底边界的运动也参与雅可比。
                normal_derivative = (np.array([[0.0, 1.0], [-1.0, 0.0]]) - np.outer(normal, edge / length)) / length
                derivative[3 * len(contacts)] = self.balance_scale * (
                    normal @ (com_jacobian - first) + offset @ normal_derivative @ (second - first)
                )
        if derivative is not None:
            write_jacobians(jacobians, self.layout.jacobian_blocks(derivative, parameters))
        return True
