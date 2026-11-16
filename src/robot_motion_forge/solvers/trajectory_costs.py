"""PyCeres 的 Python 残差及解析雅可比，支持非均匀时间步与 SO(3)。"""

from collections.abc import Sequence

import numpy as np
import pyceres

from robot_motion_forge.robot.model import FloatArray, Robot, rotation

type Parameters = Sequence[FloatArray]


type Jacobians = Sequence[FloatArray | None] | None


def _skew(vector: FloatArray) -> FloatArray:
    x, y, z = vector
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)


def inverse_right_jacobian(vector: FloatArray) -> FloatArray:
    angle2 = float(vector @ vector)
    cross = _skew(vector)
    # J_r^{-1}(v) = I + [v]/2 + (1 - theta/2*cot(theta/2))/theta^2 * [v]^2。
    # Log(SO3) 的角度在 [0, pi]；零附近用级数避免相消。
    coefficient = (
        1 / 12 + angle2 / 720 + angle2**2 / 30240
        if angle2 < 1e-8
        else (1 - np.sqrt(angle2) / (2 * np.tan(np.sqrt(angle2) / 2))) / angle2
    )
    return np.eye(3) + 0.5 * cross + coefficient * (cross @ cross)


def quaternion_lift(value: FloatArray) -> FloatArray:
    # 单位 wxyz 四元数：dq = 1/2 * [-v^T; wI + [v]] dtheta（右增量）。
    # 这里返回其切空间伪逆；Ceres 再乘自己的流形 PlusJacobian。
    return 2 * np.column_stack((-value[1:], value[0] * np.eye(3) - _skew(value[1:])))


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

    def pack(self, joints: FloatArray, positions: FloatArray, quaternions: FloatArray) -> list[FloatArray]:
        blocks = [joints] if self.robot.joint_names else []
        if self.robot.floating:
            blocks.extend([positions, quaternions])
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


class LinearCost(pyceres.CostFunction):
    """用线性组合表达参考值、速度和非均匀时间步的加速度残差。"""

    def __init__(self, size: int, coefficients: Sequence[float], target: FloatArray | None = None) -> None:
        super().__init__()
        self.set_num_residuals(size)
        self.set_parameter_block_sizes([size] * len(coefficients))
        self.coefficients = tuple(coefficients)
        self.target = np.zeros(size) if target is None else target.copy()
        self.derivatives = tuple(coefficient * np.eye(size) for coefficient in coefficients)

    def Evaluate(self, parameters: Parameters, residuals: FloatArray, jacobians: Jacobians) -> bool:
        residuals[:] = -self.target
        for coefficient, value in zip(self.coefficients, parameters, strict=True):
            residuals[:] += coefficient * value
        write_jacobians(jacobians, self.derivatives)
        return True


class RotationReferenceCost(pyceres.CostFunction):
    def __init__(self, target: FloatArray, weight: float) -> None:
        super().__init__()
        self.set_num_residuals(3)
        self.set_parameter_block_sizes([4])
        self.target_inverse = rotation(target).inv()
        self.weight = weight

    def Evaluate(self, parameters: Parameters, residuals: FloatArray, jacobians: Jacobians) -> bool:
        value = parameters[0]
        error = (self.target_inverse * rotation(value)).as_rotvec()
        residuals[:] = self.weight * error
        if jacobians is not None:
            derivative = self.weight * inverse_right_jacobian(error) @ quaternion_lift(value)
            write_jacobians(jacobians, [derivative])
        return True


class RotationDifferenceCost(pyceres.CostFunction):
    """用相邻旋转对数的线性组合表达角速度和角加速度残差。"""

    def __init__(self, coefficients: Sequence[float]) -> None:
        super().__init__()
        self.set_num_residuals(3)
        self.set_parameter_block_sizes([4] * (len(coefficients) + 1))
        self.coefficients = tuple(coefficients)

    def Evaluate(self, parameters: Parameters, residuals: FloatArray, jacobians: Jacobians) -> bool:
        orientations = rotation(np.asarray(parameters))
        errors = (orientations[:-1].inv() * orientations[1:]).as_rotvec()
        residuals[:] = np.asarray(self.coefficients) @ errors
        if jacobians is not None:
            derivatives = [np.zeros((3, 3)) for _ in parameters]
            for index, (coefficient, error) in enumerate(zip(self.coefficients, errors, strict=True)):
                derivatives[index] -= coefficient * inverse_right_jacobian(-error)
                derivatives[index + 1] += coefficient * inverse_right_jacobian(error)
            write_jacobians(
                jacobians,
                [block @ quaternion_lift(value) for block, value in zip(derivatives, parameters, strict=True)],
            )
        return True
