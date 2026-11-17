"""轨迹数据、姿态插值与采样有效性检查。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Slerp

from robot_motion_forge.robot.model import FloatArray, Robot
from robot_motion_forge.rotations import rotation_from_wxyz, rotation_to_wxyz

if TYPE_CHECKING:
    from .project import Project


@dataclass(frozen=True)
class Trajectory:
    times: FloatArray
    qpos: FloatArray
    source: str = "interpolated"
    converged: bool = False
    message: str = ""

    def __post_init__(self) -> None:
        # 节点由轨迹独立持有且只读，插值器可在整个轨迹生命周期内复用。
        times = np.array(self.times, dtype=float, copy=True)
        qpos = np.array(self.qpos, dtype=float, copy=True)
        times.flags.writeable = qpos.flags.writeable = False
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "qpos", qpos)

    @cached_property
    def _qpos_curve(self) -> PchipInterpolator:
        return PchipInterpolator(self.times, self.qpos, axis=0)

    @cached_property
    def _rotation_curve(self) -> Slerp:
        return Slerp(self.times, rotation_from_wxyz(self.qpos[:, 3:7]))

    def sample(self, robot: Robot, times: ArrayLike) -> FloatArray:
        """返回独立可写的姿态数组，查询时刻截取到节点覆盖范围内。"""
        query = np.clip(np.asarray(times, dtype=float), self.times[0], self.times[-1])
        if len(self.times) == 1:
            return np.repeat(self.qpos[:1], len(query), axis=0)
        result = self._qpos_curve(query)
        if robot.floating:
            result[:, 3:7] = rotation_to_wxyz(self._rotation_curve(query))
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
