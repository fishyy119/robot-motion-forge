"""工程内容、时间编排与求解输入指纹。"""

import hashlib
import json
from itertools import pairwise
from typing import Self

import mujoco as mj
import numpy as np
from pydantic import Field, model_validator

from robot_motion_forge.robot.model import FloatArray, Robot

from .trajectory import ReferenceTrajectory
from .types import FrameTask, Keyframe, MotionModel, Name, SolverSettings


def _loop_conflicts(robot: Robot, start: FloatArray, end: FloatArray) -> list[str]:
    # 浮动基座比较旋转差，等价的正负四元数可以闭环。容差单位为 m 或 rad。
    delta = robot.difference(start, end)
    conflicts: list[str] = []
    if robot.floating:
        for label, vector, unit in (("基座位置", delta[:3], "m"), ("基座朝向", delta[3:6], "rad")):
            if (error := np.linalg.norm(vector)) > 1e-6:
                conflicts.append(f"{label} {error:.4g} {unit}")
    for index in np.flatnonzero(abs(delta[robot.vadr]) > 1e-6):
        unit = "m" if robot.model.jnt_type[robot.joint_ids[index]] == mj.mjtJoint.mjJNT_SLIDE else "rad"
        conflicts.append(f"{robot.joint_names[index]} {abs(delta[robot.vadr[index]]):.4g} {unit}")
    return conflicts


class Project(MotionModel):
    duration: float = Field(default=2.0, gt=0)
    fps: float = Field(default=30.0, ge=1, le=1000)
    keyframes: list[Keyframe] = Field(min_length=1)
    settings: SolverSettings = Field(default_factory=SolverSettings)
    frame_tasks: dict[Name, FrameTask] = Field(default_factory=dict[str, FrameTask])

    @model_validator(mode="after")
    def check_timeline(self) -> Self:
        times = sorted(key.time for key in self.keyframes)
        if times[-1] > self.duration:
            raise ValueError("关键帧时间必须在动作范围内")
        if any(later - earlier < 1e-8 for earlier, later in pairwise(times)):
            raise ValueError("两个关键帧不能位于相同时间")
        if len({key.id for key in self.keyframes}) != len(self.keyframes):
            raise ValueError("工程包含重复 ID")
        return self

    @classmethod
    def new(cls, robot: Robot) -> "Project":
        qpos = robot.home.tolist()
        return cls(
            keyframes=[
                Keyframe(time=0.0, qpos=qpos.copy(), name="起始"),
                Keyframe(time=2.0, qpos=qpos.copy(), name="结束"),
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

    @property
    def uses_frame_tasks(self) -> bool:
        settings = self.settings
        return any(task.mode is not None and task.weight > 0 for task in self.frame_tasks.values()) and (
            settings.frame_reference > 0 or settings.frame_velocity > 0 or settings.frame_acceleration > 0
        )

    def solve_fingerprint(self) -> str:
        content: dict[str, object] = {
            "solver_version": 7,
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
            # 补齐两端的保持姿态，预览和求解参考沿用同一组作者节点。
            covered = np.unique(np.r_[0.0, times, self.duration])
            return ReferenceTrajectory(
                covered, poses[np.searchsorted(times, covered).clip(0, len(keys) - 1)], terminal_condition="rest"
            )

        start, end = keys[0], keys[-1]
        at_start, at_end = start.time == 0, end.time == self.duration
        if at_start and at_end and start.locked and end.locked:
            if conflicts := _loop_conflicts(robot, poses[0], poses[-1]):
                if preview:
                    # 编辑中保留尚未对齐的锁定姿态；求解时报告具体冲突。
                    return ReferenceTrajectory(times, poses)
                raise ValueError(
                    f"循环首尾关键帧均已锁定，姿态差异：{'、'.join(conflicts)}。请对齐首尾姿态或解锁其中一帧。"
                )

        if at_start and (start.locked or not at_end):
            seam = poses[0]
        elif at_end and (end.locked or not at_start):
            seam = poses[-1]
        else:
            # 两端均可调整时取中点；接缝位于两关键帧之间时按实际时距分配。
            before = self.duration - end.time
            fraction = 0.5 if at_start else before / (before + start.time)
            seam = robot.integrate(poses[-1], fraction * robot.difference(poses[-1], poses[0]))
        inner = (times > 0) & (times < self.duration)
        return ReferenceTrajectory(
            np.r_[0.0, times[inner], self.duration], np.vstack((seam, poses[inner], seam)), terminal_condition="loop"
        )

    def output_times(self) -> FloatArray:
        # 保持严格等间隔；非帧对齐时长的最后不足一帧区间不导出。
        return np.arange(int(np.floor(self.duration * self.fps + 1e-9)) + 1) / self.fps
