"""可版本化工程读写。JSON 不包含模型或网格，也不使用 pickle 恢复工程。"""

import json
from copy import deepcopy
from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np
from pydantic import Field

from robot_motion_forge.motion.draft import Draft
from robot_motion_forge.motion.project import Project
from robot_motion_forge.motion.trajectory import Trajectory, validate_trajectory
from robot_motion_forge.motion.types import JointNames, MotionModel, Name
from robot_motion_forge.robot.model import Robot

from .atomic import atomic_json


class SavedDraft(MotionModel):
    qpos: list[float]
    cursor: float = Field(ge=0)
    selected_key: Name | None
    locked_joints: JointNames


class SavedSolution(MotionModel):
    fingerprint: Name
    times: list[float]
    qpos: list[list[float]]
    source: str
    converged: bool
    message: str


class ProjectFile(MotionModel):
    schema_version: Literal[9]
    joint_names: JointNames
    project: Project
    draft: SavedDraft
    solution: SavedSolution | None = None


class LoadedProject(NamedTuple):
    project: Project
    draft: Draft
    trajectory: Trajectory | None


def save_project(
    path: Path, robot: Robot, project: Project, draft: Draft, trajectory: Trajectory | None = None
) -> None:
    project = project.validated(robot)
    draft = deepcopy(draft)
    draft.validate(robot, project)
    saved = None
    if trajectory is not None:
        validate_trajectory(robot, project, trajectory)
        saved = SavedSolution(
            fingerprint=project.solve_fingerprint(),
            times=trajectory.times.tolist(),
            qpos=trajectory.qpos.tolist(),
            source=trajectory.source,
            converged=trajectory.converged,
            message=trajectory.message,
        )
    document = ProjectFile(
        schema_version=9,
        joint_names=robot.joint_names.copy(),
        project=project,
        # 显式保存选中帧，保留同一时刻尚未入帧的不同草稿姿态。
        draft=SavedDraft(
            qpos=draft.qpos.tolist(),
            cursor=draft.cursor,
            selected_key=draft.selected_key,
            locked_joints=sorted(draft.locked_joints),
        ),
        solution=saved,
    )
    atomic_json(path.expanduser().resolve(), document.model_dump(exclude={"solution"} if saved is None else set()))


def load_project(path: Path, robot: Robot) -> LoadedProject:
    """读取并校验工程结构；姿态与当前模型的匹配在加载时检查。"""
    raw: object = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    document = ProjectFile.model_validate(raw)
    if document.joint_names != robot.joint_names:
        raise ValueError("工程的有序关节列表与当前机器人不一致")
    project = document.project.validated(robot)
    draft = Draft(
        qpos=np.asarray(document.draft.qpos, dtype=float),
        cursor=document.draft.cursor,
        selected_key=document.draft.selected_key,
        locked_joints=set(document.draft.locked_joints),
    )
    draft.validate(robot, project)
    saved = document.solution
    trajectory = None
    #! 此处禁止添加过宽泛的门禁，比如具体的偏移定义等
    if saved is not None and saved.fingerprint == project.solve_fingerprint():
        trajectory = Trajectory(
            np.asarray(saved.times, dtype=float),
            np.asarray(saved.qpos, dtype=float),
            saved.source,
            saved.converged,
            saved.message,
        )
        validate_trajectory(robot, project, trajectory)
    return LoadedProject(project=project, draft=draft, trajectory=trajectory)
