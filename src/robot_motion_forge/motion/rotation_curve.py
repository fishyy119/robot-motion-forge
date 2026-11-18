"""使用共享角速度连接相邻姿态的球面三次 Bézier 曲线。"""

from itertools import pairwise

import numpy as np
from scipy.spatial.transform import Rotation

from robot_motion_forge.robot.model import FloatArray


def _slerp(start: Rotation, end: Rotation, fraction: FloatArray) -> Rotation:
    """逐对沿最短弧插值，各对使用各自的时间进度。"""
    return Rotation.from_rotvec(fraction[:, None] * (end * start.inv()).as_rotvec()) * start


class RotationBezier:
    def __init__(self, times: FloatArray, orientations: Rotation) -> None:
        self.times = times
        self.orientations = orientations
        dt = np.diff(times)
        # 空间角速度用任务参考轴表达，相邻区间的转轴可直接比较。
        secants = (orientations[1:] * orientations[:-1].inv()).as_rotvec() / dt[:, None]
        speeds = np.linalg.norm(secants, axis=1)
        tangents = np.zeros((len(times), 3))
        tangents[0], tangents[-1] = secants[0], secants[-1]
        for i in range(1, len(times) - 1):
            before, after = speeds[i - 1], speeds[i]
            if before == 0 or after == 0:
                continue
            # PCHIP 加权调和平均给出速率，两单位转轴的平均使折返减速到零。
            left, right = dt[i - 1], dt[i]
            w_left, w_right = left + 2 * right, 2 * left + right
            speed = (w_left + w_right) / (w_left / before + w_right / after)
            tangents[i] = 0.5 * speed * (secants[i - 1] / before + secants[i] / after)

        # 三次 Bézier 的端点角速度是控制弧旋转向量的 3 / dt 倍。
        self.outgoing = Rotation.from_rotvec(dt[:, None] * tangents[:-1] / 3) * orientations[:-1]
        self.incoming = Rotation.from_rotvec(-dt[:, None] * tangents[1:] / 3) * orientations[1:]

    def __call__(self, query: FloatArray) -> Rotation:
        i = np.searchsorted(self.times, query, side="right").clip(1, len(self.times) - 1) - 1
        fraction = (query - self.times[i]) / (self.times[i + 1] - self.times[i])
        controls = [self.orientations[i], self.outgoing[i], self.incoming[i], self.orientations[i + 1]]
        # 球面 de Casteljau：四个控制朝向经过三层 SLERP 得到曲线上一点。
        for _ in range(3):
            controls = [_slerp(start, end, fraction) for start, end in pairwise(controls)]
        return controls[0]
