"""从作者关键帧的正向运动学生成末端位置与 SO(3) 朝向参考曲线。"""

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import RotationSpline

from robot_motion_forge.robot.model import FloatArray, quaternion, rotation
from robot_motion_forge.robot.task_frame import FrameState, TaskFrame

from .types import Keyframe


class TaskReference:
    def __init__(self, frame: TaskFrame, keyframes: list[Keyframe]) -> None:
        keys = sorted(keyframes, key=lambda key: key.time)
        self.times = np.asarray([key.time for key in keys])
        states = [frame.state(np.asarray(key.qpos)) for key in keys]
        self.positions = np.asarray([state.position for state in states])
        self.orientations = rotation(np.asarray([quaternion(state.orientation) for state in states]))
        self.position_curve = PchipInterpolator(self.times, self.positions, axis=0) if len(keys) > 1 else None
        self.rotation_curve = RotationSpline(self.times, self.orientations) if len(keys) > 1 else None

    def sample(self, times: FloatArray) -> list[FrameState]:
        query = np.clip(times, self.times[0], self.times[-1])
        if self.position_curve is None or self.rotation_curve is None:
            return [FrameState(self.positions[0].copy(), self.orientations[0]) for _ in query]
        positions = self.position_curve(query)
        orientations = self.rotation_curve(query)
        return [FrameState(positions[index], orientations[index]) for index in range(len(query))]
