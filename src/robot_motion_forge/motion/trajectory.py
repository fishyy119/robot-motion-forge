"""轨迹数据、姿态插值与采样有效性检查。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Slerp

from robot_motion_forge.robot.model import FloatArray, Robot, quaternion, rotation

if TYPE_CHECKING:
    from .project import Project


@dataclass
class Trajectory:
    times: FloatArray
    qpos: FloatArray
    source: str = "interpolated"
    converged: bool = False
    message: str = ""

    def sample(self, robot: Robot, times: ArrayLike) -> FloatArray:
        return interpolate(robot, self.times, self.qpos, np.asarray(times, dtype=float))


def interpolate(robot: Robot, times: ArrayLike, qpos: ArrayLike, query: ArrayLike) -> FloatArray:
    times, qpos = np.asarray(times, dtype=float), np.asarray(qpos, dtype=float)
    query = np.clip(np.asarray(query), times[0], times[-1])
    if len(times) == 1:
        return np.repeat(qpos[:1], len(query), axis=0)
    result = PchipInterpolator(times, qpos, axis=0)(query)
    if robot.floating:
        result[:, 3:7] = quaternion(Slerp(times, rotation(qpos[:, 3:7]))(query))
    return result


def validate_trajectory(robot: Robot, project: Project, trajectory: Trajectory) -> None:
    times = trajectory.times
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("轨迹必须包含至少两个严格递增的有限采样时刻")
    if abs(times[0]) > 1e-8 or abs(times[-1] - project.duration) > 1e-8:
        raise ValueError("轨迹必须覆盖完整动作时长")
    if trajectory.qpos.shape != (len(times), robot.model.nq):
        raise ValueError("轨迹 qpos 维度与模型不一致")
    for qpos in trajectory.qpos:
        robot.validate_qpos(qpos)
