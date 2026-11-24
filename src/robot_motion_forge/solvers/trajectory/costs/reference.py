"""关键帧姿态偏好、控制坐标系参考采样与跟踪代价。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pyceres
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation

from robot_motion_forge.motion.interpolation import RotationHermite
from robot_motion_forge.robot.closure import ClosureError
from robot_motion_forge.utils.array_types import FloatArray
from robot_motion_forge.utils.rotations import inverse_right_jacobian, rotation_from_wxyz

from ..parameters import Jacobians, Parameters, PoseLayout, quaternion_lift, write_jacobians

if TYPE_CHECKING:
    from robot_motion_forge.motion.trajectory import ReferenceTrajectory
    from robot_motion_forge.robot.definition import ControlMode
    from robot_motion_forge.robot.kinematics import ControlFrameKinematics


@dataclass(frozen=True)
class FrameReference:
    position: FloatArray | None
    orientation: Rotation | None


def sample_frame_reference(
    frame: ControlFrameKinematics, keyframes: ReferenceTrajectory, times: FloatArray, mode: ControlMode
) -> list[FrameReference]:
    """从作者关键帧的正向运动学结果插值启用的位置与 SO(3) 朝向分量。"""
    knot_times, poses = keyframes.interpolation_knots
    states = [frame.state(pose) for pose in poses]
    query = np.clip(times, keyframes.times[0], keyframes.times[-1])
    positions = None
    if mode != "orientation":
        points = np.stack([state.position for state in states])
        positions = PchipInterpolator(knot_times, points, axis=0)(query)
        if frame.reference is None and keyframes.terminal_condition == "translation_loop":
            # 世界坐标下的 XY 随基座平移，跨周期补点只用于高度与朝向等循环分量。
            positions[:, :2] = PchipInterpolator(keyframes.times, points[1:-1, :2], axis=0)(query)
    orientations = None
    if mode != "position":
        rotation_curve = RotationHermite(knot_times, Rotation.concatenate([state.orientation for state in states]))
        orientations = rotation_curve(query)
    return [
        FrameReference(
            positions[index] if positions is not None else None,
            orientations[index] if orientations is not None else None,
        )
        for index in range(len(query))
    ]


class LinearReferenceCost(pyceres.CostFunction):
    """欧氏空间的参考残差。"""

    def __init__(self, target: FloatArray, weight: float) -> None:
        super().__init__()
        size = len(target)
        self.set_num_residuals(size)
        self.set_parameter_block_sizes([size])
        self.target, self.weight = target.copy(), weight
        self.derivative = weight * np.eye(size)

    def Evaluate(self, parameters: Parameters, residuals: FloatArray, jacobians: Jacobians) -> bool:
        residuals[:] = self.weight * (parameters[0] - self.target)
        write_jacobians(jacobians, [self.derivative])
        return True


class RotationReferenceCost(pyceres.CostFunction):
    def __init__(self, target: FloatArray, weight: float) -> None:
        super().__init__()
        self.set_num_residuals(3)
        self.set_parameter_block_sizes([4])
        self.target_inverse = rotation_from_wxyz(target).inv()
        self.weight = weight

    def Evaluate(self, parameters: Parameters, residuals: FloatArray, jacobians: Jacobians) -> bool:
        value = parameters[0]
        error = (self.target_inverse * rotation_from_wxyz(value)).as_rotvec()
        residuals[:] = self.weight * error
        if jacobians is not None:
            derivative = self.weight * inverse_right_jacobian(error) @ quaternion_lift(value)
            write_jacobians(jacobians, [derivative])
        return True


class FrameReferenceCost(pyceres.CostFunction):
    def __init__(
        self, layout: PoseLayout, frame: ControlFrameKinematics, target: FrameReference, weight: float
    ) -> None:
        super().__init__()
        self.layout, self.frame = layout, frame
        self.target_position = target.position
        self.target_inverse = target.orientation.inv() if target.orientation is not None else None
        self.size = 3 * (int(self.target_position is not None) + int(self.target_inverse is not None))
        # 位置按 1 cm、旋转按 1° 归一化；weight 是平方根权重。
        self.position_scale = weight / 0.01
        self.orientation_scale = weight / np.deg2rad(1.0)
        self.set_num_residuals(self.size)
        self.set_parameter_block_sizes(self.layout.sizes)

    def Evaluate(self, parameters: Parameters, residuals: FloatArray, jacobians: Jacobians) -> bool:
        try:
            pose, lift = self.layout.reconstruct(parameters, jacobian=jacobians is not None)
        except ClosureError:
            return False
        state = self.frame.state(pose, jacobian=jacobians is not None)
        derivative = np.zeros((self.size, self.frame.robot.model.nv)) if jacobians is not None else None
        row = 0
        if self.target_position is not None:
            residuals[:3] = self.position_scale * (state.position - self.target_position)
            if derivative is not None:
                assert state.jacobian is not None
                derivative[:3] = self.position_scale * state.jacobian[:3]
            row = 3
        if self.target_inverse is not None:
            error = (self.target_inverse * state.orientation).as_rotvec()
            residuals[row : row + 3] = self.orientation_scale * error
            if derivative is not None:
                assert state.jacobian is not None
                derivative[row : row + 3] = (
                    self.orientation_scale
                    * inverse_right_jacobian(error)
                    @ state.orientation.as_matrix().T
                    @ state.jacobian[3:]
                )
        if derivative is not None:
            write_jacobians(jacobians, self.layout.jacobian_blocks(derivative, parameters, lift=lift))
        return True
