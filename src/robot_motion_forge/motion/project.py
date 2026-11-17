"""工程内容、时间编排与求解输入指纹。"""

import hashlib
import json
from typing import Self

import numpy as np
from pydantic import Field, model_validator

from robot_motion_forge.robot.model import FloatArray, Robot

from .trajectory import Trajectory
from .types import FrameTask, Keyframe, MotionModel, Name, SolverSettings


class Project(MotionModel):
    duration: float = Field(default=2.0, gt=0)
    fps: float = Field(default=30.0, ge=1, le=1000)
    keyframes: list[Keyframe] = Field(min_length=1)
    settings: SolverSettings = Field(default_factory=SolverSettings)
    frame_tasks: dict[Name, FrameTask] = Field(default_factory=dict[str, FrameTask])

    @model_validator(mode="after")
    def check_timeline(self) -> Self:
        times: list[float] = []
        identifiers: list[str] = []
        for key in self.keyframes:
            if key.time > self.duration:
                raise ValueError("关键帧时间必须在动作范围内")
            if any(abs(key.time - time) < 1e-8 for time in times):
                raise ValueError("两个关键帧不能位于相同时间")
            times.append(key.time)
            identifiers.append(key.id)
        if len(set(identifiers)) != len(identifiers):
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
            frame_tasks={name: FrameTask(mode=frame.trajectory_mode) for name, frame in robot.frames.items()},
        )

    def validated(self, robot: Robot) -> Self:
        project = type(self).model_validate(self)
        project.validate_robot(robot)
        return project

    def validate_robot(self, robot: Robot) -> None:
        """依赖当前模型的检查；字段与时间关系由 Pydantic 校验。"""
        for key in self.keyframes:
            key.qpos = robot.validate_qpos(key.qpos).tolist()
        for name, task in self.frame_tasks.items():
            frame = robot.frame(name)
            if task.reference_body is not None:
                robot.body_id(task.reference_body)
                if task.reference_body == frame.body:
                    raise ValueError(f"坐标系 {name} 的参考连杆不能是自身连杆")
        if self.settings.foot_support and not robot.support_polygons:
            raise ValueError("机器人未配置支撑轮廓")

    @property
    def uses_frame_tasks(self) -> bool:
        settings = self.settings
        return any(task.mode is not None and task.weight > 0 for task in self.frame_tasks.values()) and (
            settings.frame_reference > 0 or settings.frame_velocity > 0 or settings.frame_acceleration > 0
        )

    def solve_fingerprint(self) -> str:
        content = {
            "duration": self.duration,
            "settings": self.settings.model_dump(),
            "frame_tasks": {name: task.model_dump() for name, task in self.frame_tasks.items()},
            "keyframes": [
                key.model_dump(exclude={"id", "name"}) for key in sorted(self.keyframes, key=lambda key: key.time)
            ],
        }
        if self.uses_frame_tasks:
            content["task_output_fps"] = self.fps
        return hashlib.sha256(json.dumps(content, sort_keys=True, allow_nan=False).encode()).hexdigest()

    def keyframe_trajectory(self) -> Trajectory:
        keys = sorted(self.keyframes, key=lambda k: k.time)
        return Trajectory(
            np.asarray([key.time for key in keys], dtype=float),
            np.asarray([key.qpos for key in keys], dtype=float),
        )

    def output_times(self) -> FloatArray:
        # 保持严格等间隔；非帧对齐时长的最后不足一帧区间不导出。
        return np.arange(int(np.floor(self.duration * self.fps + 1e-9)) + 1) / self.fps
