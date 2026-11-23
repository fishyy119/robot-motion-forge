"""统一重采样与 RL 数据导出。PKL 精确遵循本地 GMR 的六个基础字段。"""

import pickle
from math import ceil

import mujoco as mj
import numpy as np
from typing_extensions import TypedDict

from robot_motion_forge.motion.project import Project
from robot_motion_forge.motion.trajectory import Trajectory, validate_trajectory
from robot_motion_forge.robot import GROUND_HEIGHT_M
from robot_motion_forge.robot.kinematics import forward_kinematics
from robot_motion_forge.robot.model import Robot
from robot_motion_forge.utils.array_types import FloatArray


class MotionPayload(TypedDict, closed=True):
    fps: float
    root_pos: FloatArray
    root_rot: FloatArray
    dof_pos: FloatArray
    local_body_pos: FloatArray
    link_body_list: list[str]


def _foot_collision_geoms(robot: Robot, model: mj.MjModel) -> list[int]:
    if not robot.floating:
        raise ValueError("基座高度修正需要浮动基座")
    if not robot.support_polygons:
        raise ValueError("基座高度修正需要配置足底支撑轮廓")
    foot_body_ids = {int(model.site_bodyid[polygon.site_id]) for polygon in robot.support_polygons.values()}
    geoms: list[int] = []
    for geom in range(model.ngeom):
        body = int(model.geom_bodyid[geom])
        if body not in foot_body_ids or model.geom_type[geom] == mj.mjtGeom.mjGEOM_HFIELD:
            continue
        # 本地 GMR 将设置碰撞掩码或位于第 3 组的几何纳入估计。
        if not (model.geom_contype[geom] or model.geom_conaffinity[geom] or model.geom_group[geom] == 3):
            continue
        geoms.append(geom)
    if not geoms:
        raise ValueError("配置的足部连杆没有可用于基座高度修正的碰撞几何")
    return geoms


def _geom_lowest_z(model: mj.MjModel, data: mj.MjData, floor: int, geom: int) -> float:
    center_z = float(data.geom_xpos[geom, 2])
    radius = float(model.geom_rbound[geom])
    # 用包围球确定查询上界，避免离地帧的距离被截断。
    distmax = abs(center_z - GROUND_HEIGHT_M) + radius + 1.0
    distance = mj.mj_geomDistance(model, data, floor, geom, distmax, None)
    # MuJoCo 无法计算该几何对的距离时，沿用 GMR 的包围球估计。
    return GROUND_HEIGHT_M + distance if distance < distmax else center_z - radius


def _tightest_height_window(sorted_heights: FloatArray, coverage: float) -> FloatArray:
    count = ceil(len(sorted_heights) * coverage)
    widths = sorted_heights[count - 1 :] - sorted_heights[: len(sorted_heights) - count + 1]
    start = int(np.argmin(widths))
    return sorted_heights[start : start + count]


def _estimate_ground_height(foot_heights: FloatArray) -> float:
    sorted_heights = np.sort(foot_heights)
    # 本地 GMR 用覆盖半数帧的最窄高度区间估计地面，宽于 3 cm 时改用四分之一帧。
    window = _tightest_height_window(sorted_heights, 0.5)
    if window[-1] - window[0] > 0.03:
        window = _tightest_height_window(sorted_heights, 0.25)
    return float(window.mean())


def _motion_payload(
    robot: Robot, project: Project, trajectory: Trajectory, *, adjust_height_to_ground: bool
) -> MotionPayload:
    validate_trajectory(robot, project.duration, trajectory)
    times = project.output_times()
    qpos = trajectory.sample(robot, times)
    # 核对重采样帧数和模型维度，再按该形状构造导出数组。
    if qpos.shape != (len(times), robot.model.nq):
        raise ValueError("导出采样的 qpos 维度与输出帧数或机器人模型不一致")
    model = robot.model_with_ground if adjust_height_to_ground else robot.model
    data = mj.MjData(model)
    body_ids = [body for body in robot.body_ids if model.body(body).name and model.body_mocapid[body] == -1]
    foot_geoms = _foot_collision_geoms(robot, model) if adjust_height_to_ground else []
    floor = int(np.flatnonzero(model.geom_bodyid == 0)[0]) if foot_geoms else -1
    foot_heights = np.empty(len(qpos)) if foot_geoms else None
    root_pos = np.empty((len(qpos), 3))
    root_quat = np.empty((len(qpos), 4))
    local_pos = np.empty((len(qpos), len(body_ids), 3))
    for index, pose in enumerate(qpos):
        forward_kinematics(model, pose, data)
        position = data.xpos[robot.root_id]
        root_pos[index] = position
        root_quat[index] = data.xquat[robot.root_id]
        local_pos[index] = (data.xpos[body_ids] - position) @ data.xmat[robot.root_id].reshape(3, 3)
        if foot_heights is not None:
            foot_heights[index] = min(_geom_lowest_z(model, data, floor, geom) for geom in foot_geoms)
    if foot_heights is not None:
        root_pos[:, 2] += GROUND_HEIGHT_M - _estimate_ground_height(foot_heights)
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
        "link_body_list": [model.body(i).name for i in body_ids],
    }


def serialize_motion(
    robot: Robot, project: Project, trajectory: Trajectory, *, adjust_height_to_ground: bool = False
) -> bytes:
    """使用已校验工程的时长与帧率生成 PKL 字节。"""
    payload = _motion_payload(robot, project, trajectory, adjust_height_to_ground=adjust_height_to_ground)
    return pickle.dumps(payload, protocol=4)
