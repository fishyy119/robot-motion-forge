"""PyCeres 的 Python 残差及解析雅可比，支持欧氏空间与 SO(3)。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pyceres

from robot_motion_forge.robot.model import FloatArray, Robot
from robot_motion_forge.rotations import inverse_right_jacobian, rotation_from_wxyz, skew

if TYPE_CHECKING:
    from robot_motion_forge.robot.definition import ControlMode
    from robot_motion_forge.robot.task_frame import FrameState, TaskFrame

type Parameters = Sequence[FloatArray]


type Jacobians = Sequence[FloatArray | None] | None


def quaternion_lift(value: FloatArray) -> FloatArray:
    # 单位 wxyz 四元数：dq = 1/2 * [-v^T; wI + [v]] dtheta（右增量）。
    # 这里返回其切空间伪逆；Ceres 再乘自己的流形 PlusJacobian。
    return 2 * np.column_stack((-value[1:], value[0] * np.eye(3) - skew(value[1:])))


def write_jacobians(jacobians: Jacobians, blocks: Sequence[FloatArray]) -> None:
    """写入 Ceres 要求的行主序参数雅可比；固定参数块不请求导数。"""
    if jacobians is not None:
        for destination, block in zip(jacobians, blocks, strict=True):
            if destination is not None:
                destination[:] = block.ravel()


class PoseLayout:
    """统一姿态参数块顺序、qpos 重建与切空间雅可比映射。"""

    def __init__(self, robot: Robot) -> None:
        self.robot = robot
        self.sizes = ([len(robot.joint_names)] if robot.joint_names else []) + ([3, 4] if robot.floating else [])

    def pack(self, pose: FloatArray) -> list[FloatArray]:
        blocks = [pose[self.robot.qadr]] if self.robot.joint_names else []
        if self.robot.floating:
            blocks.extend([pose[:3].copy(), pose[3:7].copy()])
        return blocks

    def unpack(self, parameters: Parameters) -> FloatArray:
        robot = self.robot
        pose = robot.home.copy()
        index = 0
        if robot.joint_names:
            pose[robot.qadr] = parameters[index]
            index += 1
        if robot.floating:
            pose[:3], pose[3:7] = parameters[index], parameters[index + 1]
        return pose

    def jacobian_blocks(self, derivative: FloatArray, parameters: Parameters) -> list[FloatArray]:
        robot = self.robot
        blocks: list[FloatArray] = []
        if robot.joint_names:
            blocks.append(derivative[:, robot.vadr])
        if robot.floating:
            blocks.extend([derivative[:, :3], derivative[:, 3:6] @ quaternion_lift(parameters[-1])])
        return blocks


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
    def __init__(self, frame: TaskFrame, mode: ControlMode, target: FrameState, weight: float) -> None:
        super().__init__()
        self.frame, self.target = frame, target
        self.track_position = mode in ("position", "pose")
        self.track_orientation = mode in ("orientation", "pose")
        self.size = 3 * (int(self.track_position) + int(self.track_orientation))
        # 位置按 1 cm、旋转按 1° 归一化；weight 是平方根权重。
        self.position_scale = weight / 0.01
        self.orientation_scale = weight / np.deg2rad(1.0)
        self.layout = PoseLayout(frame.robot)
        self.set_num_residuals(self.size)
        self.set_parameter_block_sizes(self.layout.sizes)

    def Evaluate(self, parameters: Parameters, residuals: FloatArray, jacobians: Jacobians) -> bool:
        state = self.frame.state(self.layout.unpack(parameters), jacobian=jacobians is not None)
        derivative = np.zeros((self.size, self.frame.robot.model.nv)) if jacobians is not None else None
        row = 0
        if self.track_position:
            residuals[:3] = self.position_scale * (state.position - self.target.position)
            if derivative is not None:
                assert state.jacobian is not None
                derivative[:3] = self.position_scale * state.jacobian[:3]
            row = 3
        if self.track_orientation:
            error = (self.target.orientation.inv() * state.orientation).as_rotvec()
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
            write_jacobians(jacobians, self.layout.jacobian_blocks(derivative, parameters))
        return True
