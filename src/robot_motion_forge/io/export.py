"""统一重采样与 RL 数据导出。PKL 精确遵循本地 GMR 的六个基础字段。"""

import pickle
from pathlib import Path

import mujoco as mj
import numpy as np
from typing_extensions import TypedDict

from robot_motion_forge.motion.project import Project
from robot_motion_forge.motion.trajectory import Trajectory, validate_trajectory
from robot_motion_forge.robot.model import FloatArray, Robot

from .atomic import atomic_write


class MotionPayload(TypedDict, closed=True):
    fps: float
    root_pos: FloatArray
    root_rot: FloatArray
    dof_pos: FloatArray
    local_body_pos: FloatArray
    link_body_list: list[str]


def motion_payload(robot: Robot, project: Project, trajectory: Trajectory) -> MotionPayload:
    project = project.validated(robot)
    validate_trajectory(robot, project, trajectory)
    times = project.output_times()
    qpos = trajectory.sample(robot, times)
    data = mj.MjData(robot.model)
    body_ids = [body for body in robot.body_ids if robot.model.body(body).name and robot.model.body_mocapid[body] == -1]
    root_pos = np.empty((len(qpos), 3))
    root_quat = np.empty((len(qpos), 4))
    local_pos = np.empty((len(qpos), len(body_ids), 3))
    for index, pose in enumerate(qpos):
        robot.forward(pose, data)
        position = data.xpos[robot.root_id]
        root_pos[index] = position
        root_quat[index] = data.xquat[robot.root_id]
        local_pos[index] = (data.xpos[body_ids] - position) @ data.xmat[robot.root_id].reshape(3, 3)
    # 连续四元数符号对依赖差分的下游更友好。
    for i in range(1, len(root_quat)):
        if np.dot(root_quat[i - 1], root_quat[i]) < 0:
            root_quat[i] *= -1
    return {
        "fps": float(project.fps),
        "root_pos": root_pos,
        "root_rot": root_quat[:, [1, 2, 3, 0]],
        "dof_pos": qpos[:, robot.qadr],
        "local_body_pos": local_pos,
        "link_body_list": [robot.model.body(i).name for i in body_ids],
    }


def serialize_motion(robot: Robot, project: Project, trajectory: Trajectory) -> bytes:
    """生成 PKL 字节，供文件导出与浏览器下载共用。"""
    return pickle.dumps(motion_payload(robot, project, trajectory), protocol=4)


def export_motion(path: Path, robot: Robot, project: Project, trajectory: Trajectory) -> None:
    path = Path(path).expanduser().resolve()
    if path.suffix != ".pkl":
        raise ValueError("动作导出扩展名必须是 .pkl")
    payload = serialize_motion(robot, project, trajectory)
    atomic_write(path, payload)
