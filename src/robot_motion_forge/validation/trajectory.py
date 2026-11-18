"""对实际重采样轨迹进行 MuJoCo 几何检查。"""

from dataclasses import dataclass

import mujoco as mj
import numpy as np

from robot_motion_forge.motion.project import Project
from robot_motion_forge.motion.trajectory import Trajectory, validate_trajectory
from robot_motion_forge.robot.model import Robot


@dataclass
class GeometryReport:
    max_ground_penetration_m: float
    max_self_penetration_m: float


def check_geometry(
    robot: Robot, project: Project, trajectory: Trajectory, *, check_fps: float = 120.0
) -> GeometryReport:
    """包含输出帧、求解节点、边界与额外密集采样；不宣称连续碰撞或动力学可行。"""
    project = project.normalized(robot)
    validate_trajectory(robot, project.duration, trajectory)
    environment = robot.environment
    uniform = np.linspace(0, project.duration, max(2, int(np.ceil(project.duration * check_fps)) + 1))
    times = np.unique(
        np.r_[
            uniform,
            project.output_times(),
            trajectory.times,
            [k.time for k in project.keyframes],
        ]
    )
    qpos = trajectory.sample(robot, times)
    data = mj.MjData(environment.model)
    ground_penetration = self_penetration = 0.0
    for pose in qpos:
        environment.forward(pose, data)
        ground_penetration = max(
            ground_penetration, np.max(environment.collision_errors(data, 0.0, kind="ground"), initial=0.0)
        )
        self_penetration = max(
            self_penetration, np.max(environment.collision_errors(data, 0.0, kind="self"), initial=0.0)
        )
    return GeometryReport(
        float(ground_penetration),
        float(self_penetration),
    )
