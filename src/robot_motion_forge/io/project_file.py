"""可版本化工程读写。JSON 不包含模型或网格，也不使用 pickle 恢复工程。"""

from copy import deepcopy
from pathlib import Path
from typing import Literal, NamedTuple, cast

import numpy as np
from pydantic import Field

from robot_motion_forge.motion.draft import Draft, PoseSource
from robot_motion_forge.motion.pose_constraints import EditConstraints
from robot_motion_forge.motion.project import Project
from robot_motion_forge.motion.trajectory import SolvedTrajectory, validate_trajectory
from robot_motion_forge.motion.types import JointNames
from robot_motion_forge.robot.model import Robot
from robot_motion_forge.utils.paths import resolve_path
from robot_motion_forge.utils.pydantic_models import Name, StrictModel

from .atomic import atomic_json


class SavedDraft(StrictModel):
    qpos: list[float]
    cursor: float = Field(ge=0)
    selected_key: Name | None
    held_key: Name | None
    constraints: EditConstraints
    pose_source: PoseSource = "previous_keyframe"


class SavedSolution(StrictModel):
    fingerprint: Name
    times: list[float]
    qpos: list[list[float]]
    qvel: list[list[float]]
    converged: bool
    message: str


class ProjectFile(StrictModel):
    schema_version: Literal[14]
    joint_names: JointNames
    project: Project
    draft: SavedDraft
    # 先核对指纹，再解析求解器生成的数据。
    solution: object = Field(default=None, exclude_if=lambda value: value is None)


class LoadedProject(NamedTuple):
    project: Project
    draft: Draft
    trajectory: SolvedTrajectory | None


def resolve_project_path(project_root: Path, relative_path: str | Path | None = None) -> Path:
    """解析工程根目录内的文件路径。"""
    if relative_path is None:
        relative_path = "untitled.motion.json"
    if isinstance(relative_path, str):
        relative_path = relative_path.strip()
    path = (project_root / relative_path).resolve()
    if project_root not in path.parents:
        raise ValueError(f"工程文件必须位于 project_root 目录内（{project_root}），实际路径为 {path}")
    return path


def save_project(
    path: Path, robot: Robot, project: Project, draft: Draft, trajectory: SolvedTrajectory | None = None
) -> None:
    project = project.normalized(robot)
    draft = deepcopy(draft)
    draft.validate(robot, project)
    saved = None
    if trajectory is not None:
        validate_trajectory(robot, project.duration, trajectory)
        saved = SavedSolution(
            fingerprint=project.solve_fingerprint(),
            times=trajectory.times.tolist(),
            qpos=trajectory.qpos.tolist(),
            qvel=trajectory.qvel.tolist(),
            converged=trajectory.converged,
            message=trajectory.message,
        )
    document = ProjectFile(
        schema_version=14,
        joint_names=robot.joint_names.copy(),
        project=project,
        # 显式保存选中帧，保留同一时刻尚未入帧的不同草稿姿态。
        draft=SavedDraft(
            qpos=draft.qpos.tolist(),
            cursor=draft.cursor,
            selected_key=draft.selected_key,
            held_key=draft.held_key,
            constraints=draft.constraints,
            pose_source=draft.pose_source,
        ),
        solution=saved,
    )
    atomic_json(resolve_path(path), document.model_dump())


def _load_solution(cache: object, robot: Robot, project: Project) -> SolvedTrajectory | None:
    # 先判定缓存是否可复用；即使缓存的数据结构损坏，工程与草稿仍可加载。
    if not isinstance(cache, dict):
        return None
    record = cast(dict[str, object], cache)
    if record.get("fingerprint") != project.solve_fingerprint():
        return None
    try:
        saved = SavedSolution.model_validate(record)
        trajectory = SolvedTrajectory(
            np.asarray(saved.times, dtype=float),
            np.asarray(saved.qpos, dtype=float),
            np.asarray(saved.qvel, dtype=float),
            saved.converged,
            saved.message,
        )
        validate_trajectory(robot, project.duration, trajectory)
    except ValueError:
        return None
    return trajectory


def load_project(path: Path, robot: Robot) -> LoadedProject:
    """读取并校验工程结构；姿态与当前模型的匹配在加载时检查。"""
    document = ProjectFile.model_validate_json(resolve_path(path).read_bytes())
    if document.joint_names != robot.joint_names:
        raise ValueError("工程的有序关节列表与当前机器人不一致")
    project = document.project.normalized(robot)
    draft = Draft(
        qpos=np.asarray(document.draft.qpos, dtype=float),
        cursor=document.draft.cursor,
        selected_key=document.draft.selected_key,
        held_key=document.draft.held_key,
        constraints=document.draft.constraints,
        pose_source=document.draft.pose_source,
    )
    draft.validate(robot, project)
    # 缓存按工程输入和轨迹数据校验，模型几何的变更由使用者自行管理。
    trajectory = _load_solution(document.solution, robot, project)
    return LoadedProject(project=project, draft=draft, trajectory=trajectory)
