"""从作者关键帧的正向运动学生成坐标系位置与 SO(3) 朝向参考曲线。"""

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation

from robot_motion_forge.robot.model import FloatArray
from robot_motion_forge.robot.task_frame import FrameState, TaskFrame

from .rotation_curve import RotationBezier
from .trajectory import ReferenceTrajectory


class TaskReference:
    def __init__(self, frame: TaskFrame, keyframes: ReferenceTrajectory) -> None:
        self.times = keyframes.times
        times, poses = keyframes.interpolation_knots
        states = [frame.state(pose) for pose in poses]
        self._first = states[0]
        self._curves = None
        if len(times) > 1:
            positions = np.stack([state.position for state in states])
            orientations = Rotation.concatenate([state.orientation for state in states])
            self._curves = PchipInterpolator(times, positions, axis=0), RotationBezier(times, orientations)

    def sample(self, times: FloatArray) -> list[FrameState]:
        query = np.clip(times, self.times[0], self.times[-1])
        if self._curves is None:
            return [FrameState(self._first.position.copy(), self._first.orientation) for _ in query]
        position_curve, rotation_curve = self._curves
        positions, orientations = position_curve(query), rotation_curve(query)
        return [FrameState(positions[index], orientations[index]) for index in range(len(query))]
