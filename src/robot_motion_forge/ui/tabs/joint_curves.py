"""按机器人配置分组展示已求解的关节位置与速度曲线。"""

from __future__ import annotations

from math import degrees
from typing import TYPE_CHECKING

import numpy as np

from ..controls import grouped_joint_names

if TYPE_CHECKING:
    from viser import GuiFolderHandle, GuiUplotHandle

    from robot_motion_forge.motion.trajectory import SolvedTrajectory
    from robot_motion_forge.robot.model import Robot
    from robot_motion_forge.utils.array_types import FloatArray

    from ..controls import Controls

_EMPTY_DATA = (np.empty(0), np.empty(0), np.empty(0))
_GRID_STROKE = "rgba(117, 132, 149, 0.50)"
_POSITION_STROKE = "#0d9488"
_VELOCITY_STROKE = "#8b5cf6"


def _axis_range(values: FloatArray, minimum_span: float) -> tuple[float, float]:
    low, high = float(values.min()), float(values.max())
    span = high - low
    center = low + span / 2
    half_span = max(minimum_span, 1.1 * span) / 2
    return center - half_span, center + half_span


class JointCurvesTab:
    def __init__(self, controls: Controls, *, robot: Robot) -> None:
        gui = controls.gui
        self._robot = robot
        self._trajectory: SolvedTrajectory | None = None
        self._cursor_time = gui.add_html("")
        self._status = gui.add_markdown("尚无求解结果" if robot.joint_names else "模型没有可绘制的关节")
        self._folders: list[GuiFolderHandle] = []
        self._plots: dict[int, tuple[GuiUplotHandle, float, float]] = {}
        joints = {name: index for index, name in enumerate(robot.joint_names)}
        for group, names in grouped_joint_names(robot).items():
            folder = gui.add_folder(group, expand_by_default=False, visible=False)
            self._folders.append(folder)
            with folder:
                for name in names:
                    index = joints[name]
                    slide = robot.is_slide_joint(index)
                    unit = "m" if slide else "°"
                    unit_factor = 1.0 if slide else degrees(1.0)
                    minimum_span = (0.001 if slide else 0.01) * unit_factor
                    plot = gui.add_uplot(
                        data=_EMPTY_DATA,
                        series=(
                            {},
                            {
                                "class": "rmf-joint-curve",
                                "scale": "y",
                                "stroke": _POSITION_STROKE,
                                "width": 2,
                                "points": {"show": False},
                            },
                            {
                                "scale": "velocity",
                                "stroke": _VELOCITY_STROKE,
                                "width": 1.5,
                                "dash": (5, 3),
                                "points": {"show": False},
                            },
                        ),
                        title=f"{name} · {unit}, {unit}/s",
                        scales={"x": {"time": False}, "y": {}, "velocity": {}},
                        axes=(
                            {"grid": {"stroke": _GRID_STROKE}},
                            {"scale": "y", "side": 3, "stroke": _POSITION_STROKE, "grid": {"stroke": _GRID_STROKE}},
                            {"scale": "velocity", "side": 1, "stroke": _VELOCITY_STROKE, "grid": {"show": False}},
                        ),
                        legend={"show": False},
                        cursor={"x": False, "y": False, "points": {"show": False}},
                        height=156,
                    )
                    self._plots[index] = plot, unit_factor, minimum_span

    def refresh(self, trajectory: SolvedTrajectory | None, cursor: float) -> None:
        # 游标只发送一个时刻；播放期间复用已采样的曲线数据。
        self._cursor_time.content = (
            f'<script type="application/json" data-rmf-curve-time>{cursor}</script>' if trajectory is not None else ""
        )
        if trajectory is self._trajectory or not self._plots:
            return
        if trajectory is None:
            for plot, _, _ in self._plots.values():
                plot.data = _EMPTY_DATA
        else:
            # 保留求解节点的准确时刻，并补充节点间的连续曲线采样。
            uniform = np.linspace(float(trajectory.times[0]), float(trajectory.times[-1]), 512)
            times = np.union1d(trajectory.times, uniform)
            positions = trajectory.sample(self._robot, times)[:, self._robot.qadr]
            velocities = trajectory.sample_joint_velocities(self._robot, times)
            for index, (plot, unit_factor, minimum_span) in self._plots.items():
                position = positions[:, index] * unit_factor
                velocity = velocities[:, index] * unit_factor
                plot.scales = {
                    "x": {"time": False},
                    "y": {"range": _axis_range(position, minimum_span)},
                    "velocity": {"range": _axis_range(velocity, minimum_span)},
                }
                plot.data = (times, position, velocity)

        for folder in self._folders:
            folder.visible = trajectory is not None
        self._status.visible = trajectory is None
        self._trajectory = trajectory
