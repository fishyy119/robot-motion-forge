"""区间三次曲线的线/角速度和线/角加速度代价。"""

from dataclasses import dataclass

import numpy as np
import pyceres
from scipy.spatial.transform import Rotation

from robot_motion_forge.motion.interpolation import hermite_basis
from robot_motion_forge.robot.closure import ClosureError
from robot_motion_forge.robot.definition import ControlMode
from robot_motion_forge.robot.kinematics import ControlFrameKinematics
from robot_motion_forge.utils.array_types import FloatArray
from robot_motion_forge.utils.rotations import (
    inverse_right_jacobian,
    log_end_tangent,
    log_end_tangent_jacobian,
    rotation_from_wxyz,
    skew,
)

from ..parameters import Jacobians, KnotVelocityMap, Parameters, PoseLayout, write_jacobians


@dataclass(frozen=True)
class _MotionState:
    position: FloatArray
    orientation: Rotation | None
    velocity: FloatArray
    pose_jacobian: FloatArray | None
    velocity_pose: FloatArray | None
    velocity_block: FloatArray | None


class SmoothnessCost(pyceres.CostFunction):
    """关节、基座与控制坐标系共用区间积分，旋转在起点的 Log 坐标中积分。"""

    def __init__(
        self,
        layout: PoseLayout,
        velocity_maps: tuple[KnotVelocityMap, KnotVelocityMap],
        step: float,
        duration: float,
        velocity_weight: float,
        acceleration_weight: float,
        *,
        frame: ControlFrameKinematics | None = None,
        mode: ControlMode = "pose",
        task_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.layout, self.velocity_maps, self.frame = layout, velocity_maps, frame
        robot = layout.robot
        self.width = len(layout.sizes) + 1
        self.dofs = robot.independent_dofs
        self.set_parameter_block_sizes([*layout.sizes, len(self.dofs)] * 2)
        self.qidx = robot.linear_qpos_indices
        self.vidx = robot.linear_qvel_indices
        self.track_position = frame is None or mode in ("position", "pose")
        self.track_orientation = robot.floating if frame is None else mode in ("orientation", "pose")
        linear_size = len(self.qidx) if frame is None else 3 * int(self.track_position)
        size = linear_size + 3 * int(self.track_orientation)
        self.set_num_residuals(3 * size)
        self.rows = slice(0 if self.track_position else 3, 6 if self.track_orientation else 3)
        order = np.r_[self.vidx, 3:6] if robot.floating else self.vidx
        self.body_jacobian = np.eye(robot.model.nv)[order] if frame is None else None

        # 三点 Gauss 积分覆盖三次曲线的速度平方与加速度平方；QR 压缩为三行残差。
        u = np.array([0.5 - np.sqrt(15) / 10, 0.5, 0.5 + np.sqrt(15) / 10])
        factor = np.sqrt(step / duration * np.array([5, 8, 5]) / 18)[:, None]
        first = np.column_stack(hermite_basis(u, step, derivative=1))
        second = np.column_stack(hermite_basis(u, step, derivative=2))
        weights = np.linalg.qr(
            np.vstack((factor * np.sqrt(velocity_weight) * first, factor * np.sqrt(acceleration_weight) * second)),
            mode="r",
        )
        scales = np.ones(size)
        if frame is not None:
            # 与坐标系参考残差一致，位置按 1 cm、旋转按 1° 归一化。
            scales[:linear_size] = np.sqrt(task_weight) / 0.01
            scales[linear_size:] = np.sqrt(task_weight) / np.deg2rad(1.0)
        self.weights: FloatArray = weights[:, :, None] * scales

    def _state(self, parameters: Parameters, mapping: KnotVelocityMap, *, jacobian: bool) -> _MotionState:
        if self.frame is None:
            pose, lift = self.layout.unpack(parameters[:-1]), None
        else:
            pose, lift = self.layout.reconstruct(parameters[:-1], jacobian=True)
        velocity, pose_slope, block_slope = mapping.decode(pose, parameters[-1], jacobian=jacobian)
        if self.frame is not None:
            # 以端点的 FK 位姿与 J(q)v 构造任务空间近似，区间内无需逐点 FK。
            state = self.frame.state(pose, jacobian=True)
            assert state.jacobian is not None
            frame_jacobian = state.jacobian if lift is None else state.jacobian @ lift
            spatial = frame_jacobian[self.rows]
            derivative = None
            if pose_slope is not None:
                derivative = spatial * pose_slope
                derivative += self.frame.velocity_pose_jacobian(pose, velocity, frame_jacobian)[self.rows]
            return _MotionState(
                state.position if self.track_position else np.empty(0),
                state.orientation if self.track_orientation else None,
                spatial @ velocity,
                spatial if jacobian else None,
                derivative,
                spatial * block_slope if block_slope is not None else None,
            )

        orientation = rotation_from_wxyz(pose[3:7]) if self.track_orientation else None
        speed = velocity[self.vidx]
        if orientation is not None:
            speed = np.r_[speed, orientation.apply(velocity[3:6])]
        spatial, derivative, speed_block = None, None, None
        if pose_slope is not None and block_slope is not None:
            assert self.body_jacobian is not None
            spatial = self.body_jacobian.copy()
            if orientation is not None:
                spatial[-3:, 3:6] = orientation.as_matrix()
            derivative = spatial * pose_slope
            if orientation is not None:
                derivative[-3:, 3:6] -= spatial[-3:, 3:6] @ skew(velocity[3:6])
            speed_block = spatial * block_slope
        return _MotionState(pose[self.qidx], orientation, speed, spatial, derivative, speed_block)

    def Evaluate(self, parameters: Parameters, residuals: FloatArray, jacobians: Jacobians) -> bool:
        left, right = parameters[: self.width], parameters[self.width :]
        try:
            a = self._state(left, self.velocity_maps[0], jacobian=jacobians is not None)
            b = self._state(right, self.velocity_maps[1], jacobian=jacobians is not None)
        except ClosureError:
            return False
        displacement, end_speed = b.position - a.position, b.velocity
        delta = None
        omega = b.velocity[-3:]
        if a.orientation is not None and b.orientation is not None:
            delta = (b.orientation * a.orientation.inv()).as_rotvec()
            displacement = np.r_[displacement, delta]
            end_speed = np.r_[b.velocity[:-3], log_end_tangent(delta, omega)]
        features = np.stack((displacement, a.velocity, end_speed))
        residuals[:] = np.einsum("kdf,df->kf", self.weights, features).ravel()
        if jacobians is None:
            return True

        assert a.pose_jacobian is not None and b.pose_jacobian is not None
        assert a.velocity_pose is not None and b.velocity_pose is not None
        assert a.velocity_block is not None and b.velocity_block is not None
        da, db = -a.pose_jacobian, b.pose_jacobian.copy()
        end_a, end_b, end_block = np.zeros_like(da), b.velocity_pose, b.velocity_block
        if delta is not None:
            log_left, log_right = inverse_right_jacobian(-delta), inverse_right_jacobian(delta)
            da[-3:] = log_right @ da[-3:]
            db[-3:] = log_left @ db[-3:]
            d_log = log_end_tangent_jacobian(delta, omega)
            end_a[-3:] = d_log @ da[-3:]
            end_b[-3:] = d_log @ db[-3:] + log_left @ end_b[-3:]
            end_block[-3:] = log_left @ end_block[-3:]
        d, v0, v1 = np.moveaxis(self.weights, 1, 0)[..., None]
        left_pose = d * da + v0 * a.velocity_pose + v1 * end_a
        right_pose = d * db + v1 * end_b
        write_jacobians(
            jacobians,
            [
                *self.layout.jacobian_blocks(left_pose.reshape(len(residuals), -1), left[:-1]),
                (v0 * a.velocity_block).reshape(len(residuals), -1)[:, self.dofs],
                *self.layout.jacobian_blocks(right_pose.reshape(len(residuals), -1), right[:-1]),
                (v1 * end_block).reshape(len(residuals), -1)[:, self.dofs],
            ],
        )
        return True
