"""围绕动作创作的关键行为保留回归覆盖。"""

import pickle
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robot_motion_forge.application.editor import Editor
from robot_motion_forge.io.export import export_motion
from robot_motion_forge.io.project import load_project
from robot_motion_forge.motion.project import Project
from robot_motion_forge.motion.trajectory import SolvedTrajectory
from robot_motion_forge.motion.types import Keyframe, PoseTarget
from robot_motion_forge.robot.model import Robot
from robot_motion_forge.rotations import rotation_from_wxyz, rotation_to_wxyz
from robot_motion_forge.solvers.trajectory import optimize
from robot_motion_forge.validation.trajectory import check_geometry


def test_ik_and_drag_preserve_valid_poses_when_targets_are_unreachable(robot: Robot) -> None:
    editor = Editor(robot)
    target_qpos = robot.home.copy()
    target_qpos[-1] = 0.7
    position, quat = robot.pose(robot.forward(target_qpos), "hand")
    result = editor.move_target(PoseTarget("hand", position.tolist(), quat.tolist()))
    assert result.success
    qpos = editor.draft_snapshot().qpos
    assert qpos[-1] == pytest.approx(0.7, abs=1e-3)
    assert np.array_equal(qpos[:7], robot.home[:7])

    previous = qpos.copy()
    result = editor.move_target(PoseTarget("hand", [100, 100, 100]))
    assert not result.success
    assert np.array_equal(editor.draft_snapshot().qpos, previous)

    editor.set_joint_locked("shoulder", True)
    target = PoseTarget("hand", [1.0, 1.0, 0.6])
    position, _ = robot.pose(robot.forward(previous), "hand")
    editor.set_anchor("torso", None)
    followed = editor.drag_target(target)
    assert followed.applied and followed.state == "projected"
    assert followed.position_error < np.linalg.norm(position - target.position)
    qpos = editor.draft_snapshot().qpos
    assert np.max(abs(robot.difference(previous, qpos)[robot.vadr])) <= np.deg2rad(3) + 1e-9
    assert np.all(qpos[robot.qadr] >= robot.lower)
    assert np.all(qpos[robot.qadr] <= robot.upper)
    assert np.array_equal(qpos[robot.qadr], previous[robot.qadr])

    editor.edit(lambda project, draft: setattr(draft, "cursor", 1.0))
    editor.capture("拖拽的实际姿态")
    captured = next(key for key in editor.project_snapshot().keyframes if key.time == 1.0)
    assert np.allclose(captured.qpos, editor.draft_snapshot().qpos)
    assert "locked_joints" not in captured.model_dump()


def test_edits_are_undoable_and_invalidate_old_solutions(robot: Robot, project: Project) -> None:
    editor = Editor(robot, project)
    editor.set_timing(0.8, 30.0)
    fingerprint = editor.solve_fingerprint
    with pytest.raises(ValueError, match="关键帧时间"):
        editor.set_timing(0.1, 30.0)
    assert editor.solve_fingerprint == fingerprint

    editor.undo()
    assert editor.duration == project.duration
    editor.redo()
    assert editor.duration == 0.8

    solution = optimize(robot, editor.project_snapshot())
    assert not editor.install_solution(solution, "obsolete")
    assert editor.install_solution(solution, editor.solve_fingerprint)
    editor.set_timing(duration=0.9)
    assert editor.trajectory is None


def test_optimization_preserves_authored_poses_times_and_joint_limits(robot: Robot, project: Project) -> None:
    project.keyframes[0].qpos[3:7] = rotation_to_wxyz(Rotation.from_euler("z", 170, degrees=True)).tolist()
    project.keyframes[-1].qpos[3:7] = rotation_to_wxyz(Rotation.from_euler("z", -170, degrees=True)).tolist()
    pose = robot.home.copy()
    pose[3:7] = rotation_to_wxyz(Rotation.from_euler("z", 180, degrees=True))
    pose[-1] = 1.5
    project.keyframes.append(Keyframe(time=0.1000000001, qpos=pose.tolist(), name="动作顶点"))

    trajectory = optimize(robot, project)
    for key in project.keyframes:
        assert key.time in trajectory.times
        sampled = trajectory.sample(robot, [key.time])[0]
        assert np.linalg.norm(robot.difference(key.qpos, sampled)) < 1e-9

    frames = trajectory.sample(robot, np.linspace(0, project.duration, 200))
    assert np.all(frames[:, robot.qadr] >= robot.lower - 1e-9)
    assert np.all(frames[:, robot.qadr] <= robot.upper + 1e-9)
    assert np.allclose(np.linalg.norm(frames[:, 3:7], axis=1), 1)
    orientations = rotation_from_wxyz(frames[:, 3:7])
    angular_path = (orientations[:-1].inv() * orientations[1:]).magnitude().sum()
    assert angular_path < np.deg2rad(30)


def test_project_restores_draft_joint_locks_and_solution(robot: Robot, project: Project, tmp_path: Path) -> None:
    editor = Editor(robot, project)
    editor.seek(0.12)
    editor.set_joint("shoulder", 0.4)
    editor.set_joint_locked("shoulder", True)
    solution = optimize(robot, editor.project_snapshot())
    editor.install_solution(solution, editor.solve_fingerprint)
    path = tmp_path / "project.motion.json"
    editor.save(path)

    saved_project, saved_draft, saved_solution = load_project(path, robot)
    restored = Editor(robot, saved_project, saved_solution, draft=saved_draft)
    draft = restored.draft_snapshot()
    assert draft.qpos[-1] == 0.4
    assert draft.cursor == 0.12
    assert draft.constraints.locked_joints == ["shoulder"]
    assert saved_solution is not None
    assert np.array_equal(saved_solution.qpos, solution.qpos)
    assert np.array_equal(saved_solution.qvel, solution.qvel)


def test_export_preserves_sampling_and_gmr_conventions(robot: Robot, project: Project, tmp_path: Path) -> None:
    for key in project.keyframes:
        key.qpos[3:7] = rotation_to_wxyz(Rotation.from_euler("z", 0.4)).tolist()
    project.keyframes[-1].qpos[-1] = 0.3
    trajectory = SolvedTrajectory(
        np.asarray([key.time for key in project.keyframes]),
        np.asarray([key.qpos for key in project.keyframes]),
        qvel=np.zeros((len(project.keyframes), robot.model.nv)),
        converged=True,
        message="",
    )
    pkl_path = tmp_path / "motion.pkl"
    export_motion(pkl_path, robot, project, trajectory)

    with pkl_path.open("rb") as stream:
        motion = pickle.load(stream)
    assert set(motion) == {
        "fps",
        "root_pos",
        "root_rot",
        "dof_pos",
        "local_body_pos",
        "link_body_list",
    }
    assert motion["fps"] == 30.0
    assert motion["dof_pos"].shape == (19, len(robot.joint_names))
    u = project.output_times() / project.duration
    assert np.allclose(motion["dof_pos"][:, -1], 0.3 * u**2 * (3 - 2 * u))
    assert np.allclose(motion["local_body_pos"][:, 0], 0)
    expected = np.asarray(project.keyframes[0].qpos)[[4, 5, 6, 3]]
    assert np.allclose(motion["root_rot"][0], expected)
    assert not pkl_path.with_suffix(".pkl.report.json").exists()


def test_geometry_checks_are_explicit_and_do_not_block_export(robot: Robot, project: Project, tmp_path: Path) -> None:
    for key in project.keyframes:
        key.qpos[2] -= 0.04
    trajectory = Editor(robot, project).reference_trajectory()
    path = tmp_path / "bad.pkl"
    report = check_geometry(robot, project, trajectory)
    assert report.max_ground_penetration_m > 0.002
    export_motion(path, robot, project, trajectory)
    assert path.exists()
