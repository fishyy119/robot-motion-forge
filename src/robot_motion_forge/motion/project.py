"""工程内容、时间编排与求解输入指纹。"""

import hashlib
import json
from itertools import pairwise
from typing import Annotated, Self
from uuid import uuid4

import numpy as np
from pydantic import Field, model_validator

from robot_motion_forge.robot.definition import ControlMode
from robot_motion_forge.robot.model import Robot
from robot_motion_forge.utils.array_types import FloatArray
from robot_motion_forge.utils.pydantic_models import Name

from .trajectory import ReferenceTrajectory
from .types import MotionModel, TerminalCondition

_DEFAULT_DURATION = 2.0

type Weight = Annotated[float, Field(ge=0)]


class Keyframe(MotionModel):
    time: float = Field(ge=0)
    qpos: list[float]
    name: str = "关键帧"
    locked: bool = True
    zero_velocity: bool = False
    id: Name = Field(default_factory=lambda: uuid4().hex[:12])


class FrameTask(MotionModel):
    mode: ControlMode | None = None
    reference_body: Name | None
    weight: Weight = 1.0


class SolverSettings(MotionModel):
    terminal_condition: TerminalCondition = "rest"
    velocity: Weight = 0.002
    acceleration: Weight = 0.002
    keyframe: Weight = 200.0
    # 碰撞权重作用于按 1 cm 归一化的距离误差平方。
    self_collision: Weight = 1.0
    ground_collision: Weight = 1.0
    collision_margin_cm: float = Field(default=0.3, ge=0)
    foot_support: bool = False
    support_position: Weight = 1.0
    support_orientation: Weight = 1.0
    balance: Weight = 1.0
    support_margin_cm: float = Field(default=0.5, ge=0)
    frame_reference: Weight = 1.0
    frame_velocity: Weight = 0.002
    frame_acceleration: Weight = 0.002
    max_iterations: int = Field(default=60, ge=1)
    solve_fps: float = Field(default=15.0, gt=0)


def _loop_conflicts(robot: Robot, start: FloatArray, end: FloatArray, *, translation_loop: bool) -> list[str]:
    # 浮动基座比较旋转差，等价的正负四元数可以闭环。容差单位为 m 或 rad。
    delta = robot.difference(start, end)
    conflicts: list[str] = []
    if robot.floating:
        position_label, position_delta = ("基座高度", delta[2:3]) if translation_loop else ("基座位置", delta[:3])
        for label, vector, unit in ((position_label, position_delta, "m"), ("基座朝向", delta[3:6], "rad")):
            if (error := np.linalg.norm(vector)) > 1e-6:
                conflicts.append(f"{label} {error:.4g} {unit}")
    for index in np.flatnonzero(abs(delta[robot.vadr]) > 1e-6):
        unit = "m" if robot.is_slide_joint(index) else "rad"
        conflicts.append(f"{robot.joint_names[index]} {abs(delta[robot.vadr[index]]):.4g} {unit}")
    return conflicts


class Project(MotionModel):
    duration: float = Field(default=_DEFAULT_DURATION, gt=0)
    fps: float = Field(default=30.0, ge=1, le=1000)
    keyframes: list[Keyframe] = Field(min_length=2)
    settings: SolverSettings = Field(default_factory=SolverSettings)
    frame_tasks: dict[Name, FrameTask] = Field(default_factory=dict[str, FrameTask])

    @model_validator(mode="after")
    def check_timeline(self) -> Self:
        times = sorted(key.time for key in self.keyframes)
        if times[-1] > self.duration:
            raise ValueError("关键帧时间必须在动作范围内")
        if times[0] != 0.0 or times[-1] != self.duration:
            raise ValueError("动作首尾必须在 0 s 和动作时长处各有一个关键帧")
        if any(later - earlier < 1e-8 for earlier, later in pairwise(times)):
            raise ValueError("两个关键帧不能位于相同时间")
        if len({key.id for key in self.keyframes}) != len(self.keyframes):
            raise ValueError("工程包含重复 ID")
        return self

    def is_boundary_key(self, key: Keyframe) -> bool:
        return key.time in (0.0, self.duration)

    @classmethod
    def new(cls, robot: Robot) -> "Project":
        qpos = robot.home.tolist()
        return cls(
            keyframes=[
                Keyframe(time=0.0, qpos=qpos.copy(), name="起始"),
                Keyframe(time=_DEFAULT_DURATION, qpos=qpos.copy(), name="结束"),
            ],
            frame_tasks={
                name: FrameTask(mode=frame.trajectory_mode, reference_body=frame.trajectory_reference_body)
                for name, frame in robot.frames.items()
                if frame.trajectory_mode is not None
            },
        )

    def normalized(self, robot: Robot) -> Self:
        """建立独立、可重复归一化的工程数据；缺少的坐标系任务视为禁用。"""
        project = type(self).model_validate(self)
        for key in project.keyframes:
            key.qpos = robot.validate_qpos(key.qpos).tolist()
        for name, task in project.frame_tasks.items():
            robot.frame_reference_id(name, task.reference_body)
        if project.settings.foot_support and not robot.support_polygons:
            raise ValueError("机器人未配置支撑轮廓")
        return project

    def solve_fingerprint(self) -> str:
        # 指纹刻意排除机器人模型，允许模型迭代后复用仍通过校验的求解缓存。
        content: dict[str, object] = {
            "solver_version": 8,
            "duration": self.duration,
            "settings": self.settings.model_dump(),
            "frame_tasks": {name: task.model_dump() for name, task in self.frame_tasks.items()},
            "keyframes": [
                key.model_dump(exclude={"id", "name"}) for key in sorted(self.keyframes, key=lambda key: key.time)
            ],
        }
        return hashlib.sha256(json.dumps(content, sort_keys=True, allow_nan=False).encode()).hexdigest()

    def keyframe_trajectory(self, robot: Robot, *, preview: bool = False) -> ReferenceTrajectory:
        """预览和求解共用作者节点及循环接缝；预览允许查看尚有冲突的锁定帧。"""
        keys = sorted(self.keyframes, key=lambda k: k.time)
        times = np.asarray([key.time for key in keys], dtype=float)
        poses = np.asarray([key.qpos for key in keys], dtype=float)
        if self.settings.terminal_condition == "rest":
            return ReferenceTrajectory(times, poses, terminal_condition="rest")

        start, end = keys[0], keys[-1]
        translation_loop = robot.floating and self.settings.terminal_condition == "translation_loop"
        if start.locked and end.locked:
            if conflicts := _loop_conflicts(robot, poses[0], poses[-1], translation_loop=translation_loop):
                if preview:
                    # 编辑中保留尚未对齐的锁定姿态；求解时报告具体冲突。
                    return ReferenceTrajectory(times, poses)
                raise ValueError(
                    f"循环首尾关键帧均已锁定，姿态差异：{'、'.join(conflicts)}。请对齐首尾姿态或解锁其中一帧。"
                )

        if start.locked:
            seam = poses[0]
        elif end.locked:
            seam = poses[-1]
        else:
            # 两端均可调整时，循环分量从独立坐标的中点开始。
            midpoint = 0.5 * robot.difference(poses[-1], poses[0])
            if robot.closure is not None:
                midpoint[robot.closure.passive] = 0.0
            seam = robot.integrate(poses[-1], midpoint)
        shared = robot.qadr
        if robot.floating:
            shared = np.r_[np.arange(2 if translation_loop else 0, 7), shared]
        poses[np.ix_([0, len(poses) - 1], shared)] = seam[shared]
        poses[0] = robot.reconstruct(poses[0])
        poses[-1] = robot.reconstruct(poses[-1]) if translation_loop else poses[0]
        return ReferenceTrajectory(times, poses, terminal_condition="translation_loop" if translation_loop else "loop")

    def output_times(self) -> FloatArray:
        # 保持严格等间隔；非帧对齐时长的最后不足一帧区间不导出。
        return np.arange(int(np.floor(self.duration * self.fps + 1e-9)) + 1) / self.fps
