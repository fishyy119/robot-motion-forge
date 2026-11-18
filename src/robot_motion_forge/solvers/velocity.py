"""节点广义速度参数和连续关节限位。"""

import numpy as np
import pyceres

from robot_motion_forge.robot.model import FloatArray, Robot


class KnotVelocity:
    """将有界关节速度映射到 [0, 1]，使相邻区间的 Bézier 控制点均满足限位。"""

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
