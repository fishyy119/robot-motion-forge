"""PyCeres 姿态与速度参数块、连续关节限位及雅可比映射。"""

from collections.abc import Sequence

import numpy as np
import pyceres

from robot_motion_forge.robot.model import Robot
from robot_motion_forge.utils.array_types import FloatArray
from robot_motion_forge.utils.rotations import skew

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
    """统一姿态参数块顺序、qpos 重建与切空间雅可比映射。浮动基座按 XY、Z、四元数分块。"""

    def __init__(self, robot: Robot) -> None:
        self.robot = robot
        self.sizes = [len(robot.joint_names)] if robot.joint_names else []
        if robot.floating:
            self.sizes.extend([2, 1, 4])
        self._manifold = pyceres.QuaternionManifold() if robot.floating else None

    def pack(self, pose: FloatArray) -> list[FloatArray]:
        blocks = [pose[self.robot.qadr]] if self.robot.joint_names else []
        if self.robot.floating:
            blocks.extend((pose[:2].copy(), pose[2:3].copy(), pose[3:7].copy()))
        return blocks

    def add_blocks(self, problem: pyceres.Problem, pose: FloatArray, *, locked: bool) -> list[FloatArray]:
        blocks = self.pack(pose)
        robot = self.robot
        if robot.joint_names:
            joints = blocks[0]
            problem.add_parameter_block(joints, self.sizes[0])
            for joint, (lower, upper) in enumerate(zip(robot.lower, robot.upper, strict=True)):
                if np.isfinite(lower):
                    problem.set_parameter_lower_bound(joints, joint, float(lower))
                if np.isfinite(upper):
                    problem.set_parameter_upper_bound(joints, joint, float(upper))
        if self._manifold is not None:
            problem.add_parameter_block(blocks[-3], self.sizes[-3])
            problem.add_parameter_block(blocks[-2], self.sizes[-2])
            problem.add_parameter_block(blocks[-1], self.sizes[-1], self._manifold)
        if locked:
            for block in blocks:
                problem.set_parameter_block_constant(block)
        return blocks

    def add_translated_endpoint(
        self, problem: pyceres.Problem, start: list[FloatArray], xy: FloatArray, *, locked: bool
    ) -> list[FloatArray]:
        """末端有独立的水平位置，与首端共用其余姿态参数块。"""
        assert self.robot.floating
        blocks = start.copy()
        blocks[-3] = xy.copy()
        problem.add_parameter_block(blocks[-3], self.sizes[-3])
        if locked:
            for block in blocks:
                problem.set_parameter_block_constant(block)
        return blocks

    def unpack(self, parameters: Parameters) -> FloatArray:
        robot = self.robot
        pose = robot.home.copy()
        if robot.joint_names:
            pose[robot.qadr] = parameters[0]
        if robot.floating:
            pose[:2], pose[2:3], pose[3:7] = parameters[-3:]
        return pose

    def jacobian_blocks(self, derivative: FloatArray, parameters: Parameters) -> list[FloatArray]:
        robot = self.robot
        blocks: list[FloatArray] = []
        if robot.joint_names:
            blocks.append(derivative[:, robot.vadr])
        if robot.floating:
            blocks.extend((derivative[:, :2], derivative[:, 2:3], derivative[:, 3:6] @ quaternion_lift(parameters[-1])))
        return blocks


class KnotVelocityMap:
    """将有限位关节的节点速度参数化为 [0, 1] 内的比例。

    用三次 Hermite 曲线估计相邻节点之间的关节运动。同一条曲线可等价表示为
    三次 Bézier 曲线。节点 q 附近的出段控制点为 q + h_out * v / 3，
    入段控制点为 q - h_in * v / 3，其中 v 是节点速度，h_out 和 h_in 是对应区间的时长。
    由凸包性质，只要每段的四个控制点均满足位置限位，段内关节位置也满足限位。
    """

    def __init__(self, robot: Robot, incoming: float | None, outgoing: float | None, *, stopped: bool) -> None:
        limited = np.isfinite(robot.lower)
        self.nv = robot.model.nv
        self.qidx, self.vidx = robot.qadr[limited], robot.vadr[limited]
        self.lower, self.upper = robot.lower[limited], robot.upper[limited]
        self.incoming = None if incoming is None else 3 / incoming
        self.outgoing = None if outgoing is None else 3 / outgoing
        self.stopped = stopped

    def _limits(self, pose: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        q = pose[self.qidx]
        if self.incoming is None:
            assert self.outgoing is not None
            slope = np.full_like(q, -self.outgoing)
            return (q - self.lower) * slope, (q - self.upper) * slope, slope, slope
        lower_in, upper_in = (q - self.upper) * self.incoming, (q - self.lower) * self.incoming
        if self.outgoing is None:
            slope = np.full_like(q, self.incoming)
            return lower_in, upper_in, slope, slope
        lower_out, upper_out = (self.lower - q) * self.outgoing, (self.upper - q) * self.outgoing
        return (
            np.maximum(lower_out, lower_in),
            np.minimum(upper_out, upper_in),
            np.where(lower_out >= lower_in, -self.outgoing, self.incoming),
            np.where(upper_out <= upper_in, -self.outgoing, self.incoming),
        )

    def add_block(self, problem: pyceres.Problem, pose: FloatArray, velocity: FloatArray) -> FloatArray:
        if self.nv == 0:
            return velocity.copy()
        if self.stopped:
            block = np.zeros(self.nv)
            problem.add_parameter_block(block, self.nv)
            problem.set_parameter_block_constant(block)
            return block
        block = velocity.copy()
        lower, upper, _, _ = self._limits(pose)
        width = upper - lower
        block[self.vidx] = np.divide(
            velocity[self.vidx] - lower, width, out=np.full_like(width, 0.5), where=width > 0
        ).clip(0, 1)
        problem.add_parameter_block(block, self.nv)
        for index in self.vidx:
            problem.set_parameter_lower_bound(block, int(index), 0.0)
            problem.set_parameter_upper_bound(block, int(index), 1.0)
        return block

    def decode(
        self, pose: FloatArray, block: FloatArray, *, jacobian: bool = False
    ) -> tuple[FloatArray, FloatArray | None, FloatArray | None]:
        if self.stopped:
            slope = np.zeros(self.nv) if jacobian else None
            return block, slope, slope
        lower, upper, d_lower, d_upper = self._limits(pose)
        fraction = block[self.vidx]
        velocity = block.copy()
        velocity[self.vidx] = (1 - fraction) * lower + fraction * upper
        if not jacobian:
            return velocity, None, None
        pose_slope, block_slope = np.zeros(self.nv), np.ones(self.nv)
        pose_slope[self.vidx] = (1 - fraction) * d_lower + fraction * d_upper
        block_slope[self.vidx] = upper - lower
        return velocity, pose_slope, block_slope
