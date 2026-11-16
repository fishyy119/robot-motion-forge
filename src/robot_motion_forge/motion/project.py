"""工程内容、时间编排与求解输入指纹。"""

import hashlib
import json
from typing import Self

import numpy as np
from numpy.typing import ArrayLike
from pydantic import Field, model_validator

from robot_motion_forge.robot.model import FloatArray, Robot

from .trajectory import Trajectory, interpolate
from .types import EndEffectorTask, Keyframe, MotionModel, Name, SolverSettings


class Project(MotionModel):
    duration: float = Field(default=2.0, gt=0)
    fps: float = Field(default=30.0, ge=1, le=1000)
    keyframes: list[Keyframe] = Field(min_length=1)
    settings: SolverSettings = Field(default_factory=SolverSettings)
    end_effector_tasks: dict[Name, EndEffectorTask] = Field(default_factory=dict[str, EndEffectorTask])

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
            end_effector_tasks={name: EndEffectorTask() for name in robot.end_effectors},
        )

    def validated(self, robot: Robot) -> Self:
        project = type(self).model_validate(self)
        project.validate_robot(robot)
        return project

    def validate_robot(self, robot: Robot) -> None:
        """依赖当前模型的检查；字段与时间关系由 Pydantic 校验。"""
        for key in self.keyframes:
            key.qpos = robot.validate_qpos(key.qpos).tolist()
        for name, task in self.end_effector_tasks.items():
            effector = robot.end_effector(name)
            if task.reference_body is not None:
                robot.body_id(task.reference_body)
                if task.reference_body == effector.body:
                    raise ValueError(f"末端 {name} 的参考连杆不能是自身连杆")
        if self.settings.foot_support:
            feet = {name: effector for name, effector in robot.end_effectors.items() if effector.foot}
            if not feet:
                raise ValueError("机器人未配置足部末端")
            for name, effector in feet.items():
                if not effector.support_polygon_cm:
                    raise ValueError(f"足部末端 {name} 未配置足底轮廓")

    @property
    def uses_end_effector_tasks(self) -> bool:
        settings = self.settings
        return any(task.mode != "none" and task.weight > 0 for task in self.end_effector_tasks.values()) and (
            settings.end_effector_reference > 0
            or settings.end_effector_velocity > 0
            or settings.end_effector_acceleration > 0
        )

    def solve_fingerprint(self) -> str:
        content = {
            "duration": self.duration,
            "settings": self.settings.model_dump(),
            "end_effector_tasks": {name: task.model_dump() for name, task in self.end_effector_tasks.items()},
            "keyframes": [
                key.model_dump(exclude={"id", "name"}) for key in sorted(self.keyframes, key=lambda key: key.time)
            ],
        }
        if self.uses_end_effector_tasks:
            content["task_output_fps"] = self.fps
        return hashlib.sha256(json.dumps(content, sort_keys=True, allow_nan=False).encode()).hexdigest()

    def seed(self, robot: Robot, times: ArrayLike) -> Trajectory:
        keys = sorted(self.keyframes, key=lambda k: k.time)
        return Trajectory(
            np.asarray(times, dtype=float),
            interpolate(robot, [k.time for k in keys], [k.qpos for k in keys], np.asarray(times)),
        )

    def output_times(self) -> FloatArray:
        # 保持严格等间隔；非帧对齐时长的最后不足一帧区间不导出。
        return np.arange(int(np.floor(self.duration * self.fps + 1e-9)) + 1) / self.fps
