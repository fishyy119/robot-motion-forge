"""PyCeres 整段轨迹优化，关键帧与关节限位为硬约束。"""

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from threading import Event
from typing import NamedTuple

import numpy as np
import pyceres

from robot_motion_forge.motion.project import Project
from robot_motion_forge.motion.support import detect_support_contacts, foot_contacts
from robot_motion_forge.motion.task_reference import TaskReference
from robot_motion_forge.motion.trajectory import Trajectory
from robot_motion_forge.robot.model import FloatArray, Robot
from robot_motion_forge.robot.task_frame import TaskFrame

from .collision import CollisionCost
from .end_effector import EndEffectorDifferenceCost, EndEffectorReferenceCost
from .support import SupportCost
from .trajectory_costs import LinearCost, PoseLayout, RotationDifferenceCost, RotationReferenceCost


class SolveCancelled(RuntimeError):
    pass


class _IterationCallback(pyceres.IterationCallback):
    """报告实际迭代代价，在迭代边界响应取消。"""

    def __init__(self, cancel: Event | None, progress: Callable[[int, float], None] | None) -> None:
        super().__init__()
        self.cancel, self.progress = cancel, progress
        self.iterations = 0

    def __call__(self, summary: pyceres.IterationSummary) -> pyceres.CallbackReturnType:
        self.iterations = int(summary.iteration)
        if self.cancel is not None and self.cancel.is_set():
            return pyceres.CallbackReturnType.SOLVER_ABORT
        if self.progress is not None:
            self.progress(self.iterations, float(summary.cost))
        if self.cancel is not None and self.cancel.is_set():
            return pyceres.CallbackReturnType.SOLVER_ABORT
        return pyceres.CallbackReturnType.SOLVER_CONTINUE


@dataclass
class _PreparedTrajectory:
    times: FloatArray
    reference: FloatArray
    free: set[int]
    layout: PoseLayout
    values: list[FloatArray]
    problem: pyceres.Problem

    def parameters(self, indices: Sequence[int]) -> list[FloatArray]:
        return [values[index] for index in indices for values in self.values]

    def qpos(self) -> FloatArray:
        result = self.reference.copy()
        for frame in self.free:
            result[frame] = self.layout.unpack(self.parameters([frame]))
        return result


class _Difference(NamedTuple):
    indices: tuple[int, ...]
    linear_coefficients: tuple[float, ...]
    rotation_coefficients: tuple[float, ...]
    weight: float


def _differences(
    times: FloatArray, duration: float, velocity: float, acceleration: float, *, weight_scale: float = 1.0
) -> Iterator[_Difference]:
    """关节、基座与末端共用非均匀时间步及按动作时长归一化的差分权重。"""
    dt = np.diff(times)
    if velocity > 0:
        for index, step in enumerate(dt):
            yield _Difference(
                (index, index + 1), (-1.0, 1.0), (1.0,), float(np.sqrt(weight_scale * velocity / (step * duration)))
            )
    if acceleration > 0:
        for index, (first, second) in enumerate(zip(dt[:-1], dt[1:], strict=True)):
            weight = (
                float(np.sqrt(weight_scale * acceleration * (first + second) / (2 * duration))) * 2 / (first + second)
            )
            yield _Difference(
                (index, index + 1, index + 2),
                (1 / first, -1 / first - 1 / second, 1 / second),
                (-1 / first, 1 / second),
                weight,
            )


def _add_end_effector_tasks(prepared: _PreparedTrajectory, robot: Robot, project: Project) -> None:
    if not project.uses_end_effector_tasks:
        return
    settings, problem, times = project.settings, prepared.problem, prepared.times

    for name, task in project.end_effector_tasks.items():
        if task.mode == "none" or task.weight <= 0:
            continue
        frame = TaskFrame(robot, name, task.reference_body)
        if settings.end_effector_reference > 0:
            targets = TaskReference(frame, project.keyframes).sample(times)
            weight = float(np.sqrt(task.weight * settings.end_effector_reference / len(times)))
            for index in sorted(prepared.free):
                problem.add_residual_block(
                    EndEffectorReferenceCost(frame, task.mode, targets[index], weight),
                    None,
                    prepared.parameters([index]),
                )
        for difference in _differences(
            times,
            project.duration,
            settings.end_effector_velocity,
            settings.end_effector_acceleration,
            weight_scale=task.weight,
        ):
            if not prepared.free.intersection(difference.indices):
                continue
            problem.add_residual_block(
                EndEffectorDifferenceCost(
                    frame,
                    task.mode,
                    position_coefficients=difference.linear_coefficients,
                    rotation_coefficients=difference.rotation_coefficients,
                    weight=difference.weight,
                ),
                None,
                prepared.parameters(difference.indices),
            )


def _prepare(robot: Robot, project: Project) -> _PreparedTrajectory:
    settings = project.settings
    # 非帧对齐关键帧作为独立节点加入，避免约束被舍入或漏掉。
    regular = np.linspace(0, project.duration, max(2, int(np.ceil(project.duration * settings.solve_fps)) + 1))
    protected = np.unique([0.0, project.duration, *(key.time for key in project.keyframes)])
    if project.uses_end_effector_tasks:
        # 启用末端任务时直接约束输出帧，避免最终逐关节重采样改变末端目标；作者时间仍优先。
        output = [time for time in project.output_times() if not np.any(np.isclose(protected, time, rtol=0, atol=1e-8))]
        protected = np.sort(np.r_[protected, output])
    # 优先保留作者给出的准确时间，移除与其几乎重合的规则节点，避免极短时间步。
    extra = [time for time in regular if not np.any(np.isclose(protected, time, rtol=0, atol=1e-8))]
    times = np.sort(np.r_[protected, extra])
    reference = project.seed(robot, times).qpos
    count = len(times)
    key_indices = {key.id: int(np.argmin(abs(times - key.time))) for key in project.keyframes}
    fixed = {key_indices[key.id] for key in project.keyframes if key.locked}
    for key in project.keyframes:
        if key.locked:
            reference[key_indices[key.id]] = key.qpos
    free = set(range(count)) - fixed
    joints = np.ascontiguousarray(reference[:, robot.qadr])
    positions = np.ascontiguousarray(reference[:, :3]) if robot.floating else np.zeros((count, 0))
    quaternions = np.ascontiguousarray(reference[:, 3:7]) if robot.floating else np.zeros((count, 0))
    # 参数数组由准备结果持有；PyCeres 问题负责保留残差和流形对象的引用。
    problem = pyceres.Problem()
    layout = PoseLayout(robot)
    prepared = _PreparedTrajectory(times, reference, free, layout, layout.pack(joints, positions, quaternions), problem)
    quaternion_manifold = pyceres.QuaternionManifold()
    for frame in range(count):
        if robot.joint_names:
            problem.add_parameter_block(joints[frame], len(robot.joint_names))
            for index, (lower, upper) in enumerate(zip(robot.lower, robot.upper, strict=True)):
                if np.isfinite(lower):
                    problem.set_parameter_lower_bound(joints[frame], index, float(lower))
                if np.isfinite(upper):
                    problem.set_parameter_upper_bound(joints[frame], index, float(upper))
        if robot.floating:
            problem.add_parameter_block(positions[frame], 3)
            problem.add_parameter_block(quaternions[frame], 4, quaternion_manifold)
        if frame in fixed:
            for value in prepared.parameters([frame]):
                problem.set_parameter_block_constant(value)

    def add_reference(frame: int, weight: float, target: FloatArray) -> None:
        if weight <= 0 or frame in fixed:
            return
        weight = float(np.sqrt(weight))
        if robot.joint_names:
            problem.add_residual_block(
                LinearCost(len(robot.joint_names), [weight], weight * target[robot.qadr]), None, [joints[frame]]
            )
        if robot.floating:
            problem.add_residual_block(LinearCost(3, [weight], weight * target[:3]), None, [positions[frame]])
            problem.add_residual_block(RotationReferenceCost(target[3:7], weight), None, [quaternions[frame]])

    for frame in sorted(free):
        add_reference(frame, settings.style / count, reference[frame])
    for key in project.keyframes:
        add_reference(key_indices[key.id], settings.keyframe, np.asarray(key.qpos))

    euclidean = prepared.values[:-1] if robot.floating else prepared.values
    for difference in _differences(times, project.duration, settings.velocity, settings.acceleration):
        if not free.intersection(difference.indices):
            continue
        coefficients = [difference.weight * coefficient for coefficient in difference.linear_coefficients]
        for values in euclidean:
            problem.add_residual_block(
                LinearCost(values.shape[1], coefficients), None, [values[index] for index in difference.indices]
            )
        if robot.floating:
            problem.add_residual_block(
                RotationDifferenceCost(
                    [difference.weight * coefficient for coefficient in difference.rotation_coefficients]
                ),
                None,
                [quaternions[index] for index in difference.indices],
            )
    environment = robot.environment
    if settings.foot_support and (
        settings.support_position > 0 or settings.support_orientation > 0 or settings.balance > 0
    ):
        data = robot.forward(reference[0])
        for frame in sorted(free):
            robot.forward(reference[frame], data)
            contacts = detect_support_contacts(foot_contacts(robot, data), nearest=True)
            problem.add_residual_block(
                SupportCost(robot, contacts, settings, count), None, prepared.parameters([frame])
            )
    if environment.collision_pairs and (settings.self_collision > 0 or settings.ground_collision > 0):
        pairs = np.asarray(environment.collision_pairs, dtype=int)
        ground = np.any(environment.model.geom_bodyid[pairs] == 0, axis=1)
        weights = np.sqrt(np.where(ground, settings.ground_collision, settings.self_collision) / count)
        for frame in sorted(free):
            problem.add_residual_block(
                CollisionCost(robot, weights, margin_cm=settings.collision_margin_cm),
                None,
                prepared.parameters([frame]),
            )
    _add_end_effector_tasks(prepared, robot, project)
    return prepared


def optimize(
    robot: Robot,
    project: Project,
    *,
    cancel: Event | None = None,
    progress: Callable[[int, float], None] | None = None,
) -> Trajectory:
    if cancel is not None and cancel.is_set():
        raise SolveCancelled("已取消求解")
    project = project.validated(robot)
    prepared = _prepare(robot, project)
    if cancel is not None and cancel.is_set():
        raise SolveCancelled("已取消求解")
    if not prepared.free or prepared.problem.num_residual_blocks() == 0:
        return Trajectory(prepared.times, prepared.reference, "optimized", True, "没有可优化的自由目标")
    callback = _IterationCallback(cancel, progress)
    options = pyceres.SolverOptions()
    options.max_num_iterations = project.settings.max_iterations
    # Python 自定义残差包含回调开销，本项目实测单线程优于多线程。
    options.num_threads = 1
    options.linear_solver_type = pyceres.LinearSolverType.SPARSE_NORMAL_CHOLESKY
    options.minimizer_progress_to_stdout = False
    options.logging_type = pyceres.LoggingType.SILENT
    options.callbacks = [callback]
    # 固定停止阈值，与已验证的 PyCeres 默认设置保持一致。
    options.function_tolerance = 1e-6
    options.gradient_tolerance = 1e-10
    options.parameter_tolerance = 1e-8
    summary = pyceres.SolverSummary()
    pyceres.solve(options, prepared.problem, summary)
    if cancel is not None and cancel.is_set():
        raise SolveCancelled("已取消求解")
    if not summary.IsSolutionUsable():
        raise RuntimeError(str(summary.message))
    qpos = prepared.qpos()
    if not np.isfinite(qpos).all():
        raise ValueError("求解器返回了非有限轨迹")
    cost = float(summary.final_cost)
    converged = summary.termination_type in (pyceres.TerminationType.CONVERGENCE, pyceres.TerminationType.USER_SUCCESS)
    message = f"迭代 {callback.iterations} · 代价 {cost:.5g}"
    if not converged:
        message = f"达到最大迭代次数 · {message}"
    return Trajectory(prepared.times, qpos, "optimized", converged, message)
