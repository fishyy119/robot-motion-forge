"""统一重采样与 RL 数据导出。PKL 精确遵循本地 GMR 的六个基础字段。"""

import pickle
from typing import cast

import jaxtyping
import mujoco as mj
import numpy as np
from typing_extensions import TypedDict

from robot_motion_forge.motion.project import Project
from robot_motion_forge.motion.trajectory import Trajectory, validate_trajectory
from robot_motion_forge.robot.model import Robot
from robot_motion_forge.utils.array_types import FloatArray


class MotionPayload(TypedDict, closed=True):
    fps: float
    root_pos: FloatArray
    root_rot: FloatArray
    dof_pos: FloatArray
    local_body_pos: FloatArray
    link_body_list: list[str]


def _validate_motion_payload(payload: MotionPayload, times: FloatArray, joint_count: int) -> None:
    # 先绑定输出时间轴的帧数，避免各数组同时缺帧时仍通过检查。
    arrays = (
        ("times", times, "frames"),
        ("root_pos", payload["root_pos"], "frames 3"),
        ("root_rot", payload["root_rot"], "frames 4"),
        ("dof_pos", payload["dof_pos"], f"frames {joint_count}"),
        ("local_body_pos", payload["local_body_pos"], f"frames {len(payload['link_body_list'])} 3"),
    )
    with jaxtyping.jaxtyped("context"):  # pyright: ignore[reportUnknownMemberType]
        for field, values, shape in arrays:
            # Float64 的形状检查由运行时生成的类实现。
            array_type = cast(type[object], jaxtyping.Float64[np.ndarray, shape])
            if not isinstance(values, array_type):
                expected = shape.replace("frames", str(len(times))).replace(" ", ", ") if field != "times" else "N,"
                raise ValueError(f"导出数组 {field} 需要 float64 和形状 ({expected})，实际为 {values.dtype} {values.shape}")


def _motion_payload(robot: Robot, project: Project, trajectory: Trajectory) -> MotionPayload:
    project = project.normalized(robot)
    validate_trajectory(robot, project.duration, trajectory)
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
    payload: MotionPayload = {
        "fps": float(project.fps),
        "root_pos": root_pos,
        "root_rot": root_quat[:, [1, 2, 3, 0]],
        "dof_pos": qpos[:, robot.qadr],
        "local_body_pos": local_pos,
        "link_body_list": [robot.model.body(i).name for i in body_ids],
    }
    _validate_motion_payload(payload, times, len(robot.joint_names))
    return payload


def serialize_motion(robot: Robot, project: Project, trajectory: Trajectory) -> bytes:
    """生成供浏览器下载的 PKL 字节。"""
    return pickle.dumps(_motion_payload(robot, project, trajectory), protocol=4)
