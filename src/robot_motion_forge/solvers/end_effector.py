"""通用末端轨迹残差：参考跟随、线/角速度与线/角加速度。"""

from collections.abc import Sequence

import numpy as np
import pyceres

from robot_motion_forge.motion.types import EndEffectorTaskMode
from robot_motion_forge.robot.model import FloatArray
from robot_motion_forge.robot.task_frame import FrameState, TaskFrame

from .trajectory_costs import Jacobians, Parameters, PoseLayout, inverse_right_jacobian, write_jacobians


class _FrameCost(pyceres.CostFunction):
    def __init__(self, frame: TaskFrame, mode: EndEffectorTaskMode, count: int, weight: float) -> None:
        super().__init__()
        self.frame, self.count = frame, count
        self.track_position = mode in ("position", "pose")
        self.track_orientation = mode in ("orientation", "pose")
        self.size = 3 * (int(self.track_position) + int(self.track_orientation))
        if self.size == 0:
            raise ValueError("禁用的末端任务不应创建优化残差")
        # 位置按 1 cm、旋转按 1° 归一化；weight 是平方根权重。
        self.position_scale = weight / 0.01
        self.orientation_scale = weight / np.deg2rad(1.0)
        self.layout = PoseLayout(frame.robot)
        self.set_num_residuals(self.size)
        self.set_parameter_block_sizes(self.layout.sizes * count)

    def _states(self, parameters: Parameters, *, jacobian: bool) -> list[FrameState]:
        width = len(self.layout.sizes)
        return [
            self.frame.state(self.layout.unpack(parameters[index * width : (index + 1) * width]), jacobian=jacobian)
            for index in range(self.count)
        ]

    def _write(self, jacobians: Jacobians, derivatives: list[FloatArray], parameters: Parameters) -> None:
        blocks: list[FloatArray] = []
        width = len(self.layout.sizes)
        for index, derivative in enumerate(derivatives):
            blocks.extend(self.layout.jacobian_blocks(derivative, parameters[index * width : (index + 1) * width]))
        write_jacobians(jacobians, blocks)


class EndEffectorReferenceCost(_FrameCost):
    def __init__(self, frame: TaskFrame, mode: EndEffectorTaskMode, target: FrameState, weight: float) -> None:
        super().__init__(frame, mode, 1, weight)
        self.target = target

    def Evaluate(self, parameters: Parameters, residuals: FloatArray, jacobians: Jacobians) -> bool:
        state = self._states(parameters, jacobian=jacobians is not None)[0]
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
            self._write(jacobians, [derivative], parameters)
        return True


class EndEffectorDifferenceCost(_FrameCost):
    def __init__(
        self,
        frame: TaskFrame,
        mode: EndEffectorTaskMode,
        *,
        position_coefficients: Sequence[float],
        rotation_coefficients: Sequence[float],
        weight: float,
    ) -> None:
        super().__init__(frame, mode, len(position_coefficients), weight)
        self.position_coefficients = tuple(position_coefficients)
        self.rotation_coefficients = tuple(rotation_coefficients)
        if len(self.rotation_coefficients) != self.count - 1:
            raise ValueError("旋转差分系数数量与采样节点不一致")

    def Evaluate(self, parameters: Parameters, residuals: FloatArray, jacobians: Jacobians) -> bool:
        states = self._states(parameters, jacobian=jacobians is not None)
        derivatives = (
            [np.zeros((self.size, self.frame.robot.model.nv)) for _ in states] if jacobians is not None else None
        )
        row = 0
        if self.track_position:
            residuals[:3] = self.position_scale * (
                np.asarray(self.position_coefficients) @ np.asarray([state.position for state in states])
            )
            if derivatives is not None:
                for index, coefficient in enumerate(self.position_coefficients):
                    jacobian = states[index].jacobian
                    assert jacobian is not None
                    derivatives[index][:3] = self.position_scale * coefficient * jacobian[:3]
            row = 3
        if self.track_orientation:
            residuals[row : row + 3] = 0.0
            for index, coefficient in enumerate(self.rotation_coefficients):
                first, second = states[index], states[index + 1]
                # 空间旋转增量统一表达在任务参考坐标轴下，跨区间相减时不混用局部坐标系。
                error = (second.orientation * first.orientation.inv()).as_rotvec()
                scale = self.orientation_scale * coefficient
                residuals[row : row + 3] += scale * error
                if derivatives is not None:
                    assert first.jacobian is not None and second.jacobian is not None
                    derivatives[index][row : row + 3] -= scale * inverse_right_jacobian(error) @ first.jacobian[3:]
                    derivatives[index + 1][row : row + 3] += (
                        scale * inverse_right_jacobian(-error) @ second.jacobian[3:]
                    )
        if derivatives is not None:
            self._write(jacobians, derivatives, parameters)
        return True
