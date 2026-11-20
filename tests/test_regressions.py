"""围绕动作创作的关键行为保留回归覆盖。"""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robot_motion_forge.application.editor import Editor
from robot_motion_forge.motion.pose_constraints import (
    ANCHOR_ORIENTATION_TOLERANCE_RAD,
    ANCHOR_POSITION_TOLERANCE_M,
    PoseTarget,
)
from robot_motion_forge.motion.project import Keyframe, Project
from robot_motion_forge.motion.trajectory import SolvedTrajectory
from robot_motion_forge.robot.model import Robot
from robot_motion_forge.solvers.trajectory.solve import optimize
from robot_motion_forge.utils.rotations import rotation_from_wxyz, rotation_to_wxyz


def test_drag_preserves_valid_poses_when_targets_are_unreachable(robot: Robot) -> None:
    editor = Editor(robot)
    shoulder = robot.qadr[robot.joint_names.index("shoulder")]
    target_qpos = robot.home.copy()
    target_qpos[shoulder] = 0.006
    position, quat = robot.frame_pose(robot.forward(target_qpos), "hand")
    result = editor.drag_target(PoseTarget("hand", position.tolist(), quat.tolist()))
    assert result.applied and result.reached
    previous = editor.draft_snapshot().qpos
    assert previous[shoulder] == pytest.approx(target_qpos[shoulder], abs=1e-3)
    anchor_delta = robot.difference(robot.home, previous)
    assert np.linalg.norm(anchor_delta[:3]) <= ANCHOR_POSITION_TOLERANCE_M
    assert np.linalg.norm(anchor_delta[3:6]) <= ANCHOR_ORIENTATION_TOLERANCE_RAD

    editor.set_joint_locked("shoulder", True)
    target = PoseTarget("hand", [1.0, 1.0, 0.6])
    blocked = editor.drag_target(target)
    assert blocked.state == "blocked"
    assert np.array_equal(editor.draft_snapshot().qpos, previous)

    position, _ = robot.frame_pose(robot.forward(previous), "hand")
    editor.set_anchor("torso", None)
    followed = editor.drag_target(target)
    assert followed.applied and followed.state == "projected"
    assert followed.position_error < np.linalg.norm(position - target.position)
    qpos = editor.draft_snapshot().qpos
    delta = robot.difference(previous, qpos)
    assert np.linalg.norm(delta[:3]) <= 0.003 + 1e-8
    assert np.linalg.norm(delta[3:6]) <= np.deg2rad(3) + 1e-7
    assert np.all(qpos[robot.qadr] >= robot.lower)
    assert np.all(qpos[robot.qadr] <= robot.upper)
    assert qpos[shoulder] == previous[shoulder]

    editor.scrub(1.0)
    editor.capture("拖拽的实际姿态")
    captured = next(key for key in editor.project_snapshot().keyframes if key.time == 1.0)
    assert np.allclose(captured.qpos, qpos)


def test_edits_are_undoable_and_invalidate_old_solutions(robot: Robot, project: Project) -> None:
    project.keyframes.insert(1, Keyframe(time=0.3, qpos=robot.home.tolist()))
    editor = Editor(robot, project)
    editor.set_timing(0.8)
    editor.seek(0.45)
    edited = editor.project_snapshot()
    with pytest.raises(ValueError, match="关键帧时间"):
        editor.set_timing(0.1, 60.0)
    assert editor.project_snapshot() == edited
    assert editor.draft_snapshot().cursor == 0.45

    editor.undo()
    assert editor.project_snapshot() == project
    editor.redo()
    assert editor.project_snapshot() == edited

    fingerprint = editor.solve_fingerprint
    solution = optimize(robot, edited)
    assert editor.install_solution(solution, fingerprint)
    editor.set_settings(velocity=0.01)
    assert editor.trajectory is None
    assert not editor.install_solution(solution, fingerprint)
    assert editor.trajectory is None


def test_optimization_preserves_authored_poses_times_and_joint_limits(robot: Robot, project: Project) -> None:
    project.keyframes[0].qpos[3:7] = rotation_to_wxyz(Rotation.from_euler("z", 170, degrees=True)).tolist()
    project.keyframes[-1].qpos[3:7] = rotation_to_wxyz(Rotation.from_euler("z", -170, degrees=True)).tolist()
    pose = robot.home.copy()
    pose[3:7] = rotation_to_wxyz(Rotation.from_euler("z", 180, degrees=True))
    shoulder = robot.joint_names.index("shoulder")
    pose[robot.qadr[shoulder]] = robot.upper[shoulder]
    project.keyframes.append(Keyframe(time=0.1000000001, qpos=pose.tolist(), name="动作顶点"))

    trajectory = optimize(robot, project)
    for key in project.keyframes:
        assert key.time in trajectory.times
        sampled = trajectory.sample(robot, [key.time])[0]
        assert np.linalg.norm(robot.difference(key.qpos, sampled)) < 1e-9
    assert np.allclose(trajectory.qvel[[0, -1]], 0.0, atol=1e-9)

    frames = trajectory.sample(robot, np.linspace(0, project.duration, 200))
    assert np.all(frames[:, robot.qadr] >= robot.lower - 1e-9)
    assert np.all(frames[:, robot.qadr] <= robot.upper + 1e-9)
    assert np.allclose(np.linalg.norm(frames[:, 3:7], axis=1), 1)
    orientations = rotation_from_wxyz(frames[:, 3:7])
    angular_path = (orientations[:-1].inv() * orientations[1:]).magnitude().sum()
    assert angular_path < np.deg2rad(30)


def test_project_restores_draft_joint_locks_and_solution(robot: Robot, project: Project, tmp_path: Path) -> None:
    shoulder = robot.qadr[robot.joint_names.index("shoulder")]
    project.keyframes[-1].qpos[shoulder] = 0.25
    editor = Editor(robot, project)
    authored = editor.project_snapshot()
    editor.seek(0.12)
    editor.set_joint("shoulder", 0.4)
    editor.set_joint_locked("shoulder", True)
    solution = optimize(robot, authored)
    assert editor.install_solution(solution, editor.solve_fingerprint)
    path = tmp_path / "project.motion.json"
    editor.save(path)

    restored = Editor.load(robot, path)
    draft = restored.draft_snapshot()
    assert draft.qpos[shoulder] == 0.4
    assert draft.cursor == 0.12
    assert draft.constraints.locked_joints == ["shoulder"]
    assert restored.project_snapshot().keyframes == authored.keyframes
    saved_solution = restored.trajectory
    assert saved_solution is not None
    assert np.array_equal(saved_solution.times, solution.times)
    assert np.array_equal(saved_solution.qpos, solution.qpos)
    assert np.array_equal(saved_solution.qvel, solution.qvel)


def test_solved_trajectory_samples_joint_positions_and_velocities(robot: Robot) -> None:
    shoulder = robot.joint_names.index("shoulder")
    knots = np.array([0.0, 0.2, 0.6])
    qpos = np.tile(robot.home, (len(knots), 1))
    qvel = np.zeros((len(knots), robot.model.nv))
    # 三次 Hermite 插值应重现给定三次多项式及其时间导数。
    qpos[:, robot.qadr[shoulder]] = knots**3 + 0.2 * knots
    qvel[:, robot.vadr[shoulder]] = 3 * knots**2 + 0.2
    trajectory = SolvedTrajectory(knots, qpos, qvel, converged=True, message="")

    times = np.array([0.0, 0.07, 0.2, 0.42, 0.6])
    positions = trajectory.sample(robot, times)[:, robot.qadr[shoulder]]
    velocities = trajectory.sample_joint_velocities(robot, times)[:, shoulder]
    np.testing.assert_allclose(positions, times**3 + 0.2 * times, rtol=0, atol=1e-12)
    np.testing.assert_allclose(velocities, 3 * times**2 + 0.2, rtol=0, atol=1e-12)
