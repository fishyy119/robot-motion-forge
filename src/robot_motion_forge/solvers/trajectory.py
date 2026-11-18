"""PyCeres 整段轨迹优化，关键帧与关节限位为硬约束。"""

from collections.abc import Callable
from dataclasses import dataclass
from threading import Event

import numpy as np
import pyceres

from robot_motion_forge.motion.project import Project
from robot_motion_forge.motion.task_reference import TaskReference
from robot_motion_forge.motion.trajectory import ReferenceTrajectory, SolvedTrajectory
from robot_motion_forge.robot.definition import ControlMode
from robot_motion_forge.robot.model import FloatArray, Robot
from robot_motion_forge.robot.support import detect_support_contacts, foot_contacts
from robot_motion_forge.robot.task_frame import TaskFrame
from robot_motion_forge.rotations import rotation_from_wxyz

from .collision import CollisionCost
from .motion_cost import MotionCost
from .support import SupportCost
from .trajectory_costs import FrameReferenceCost, LinearReferenceCost, PoseLayout, RotationReferenceCost
from .velocity import KnotVelocity


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


@dataclass(frozen=True)
class _SolveNode:
    pose: list[FloatArray]
    locked: bool
    velocity_mapping: KnotVelocity
    velocity: FloatArray


@dataclass(frozen=True)
class _PreparedTrajectory:
    times: FloatArray
    nodes: list[_SolveNode]
    layout: PoseLayout
    problem: pyceres.Problem

    def result(self, converged: bool, message: str) -> SolvedTrajectory:
        qpos = np.stack([self.layout.unpack(node.pose) for node in self.nodes])
        qvel = np.stack(
            [node.velocity_mapping.decode(pose, node.velocity)[0] for node, pose in zip(self.nodes, qpos, strict=True)]
        )
        if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
            raise ValueError("求解器返回了非有限轨迹")
        return SolvedTrajectory(self.times, qpos, qvel, converged, message)


def _add_motion_costs(
    prepared: _PreparedTrajectory,
    velocity: float,
    acceleration: float,
    *,
    frame: TaskFrame | None = None,
    mode: ControlMode = "pose",
    task_weight: float = 1.0,
) -> None:
    if velocity == 0 and acceleration == 0:
        return
    duration = float(prepared.times[-1])
    for index, step in enumerate(np.diff(prepared.times)):
        left, right = prepared.nodes[index : index + 2]
        prepared.problem.add_residual_block(
            MotionCost(
                prepared.layout,
                (left.velocity_mapping, right.velocity_mapping),
                float(step),
                duration,
                velocity,
                acceleration,
                frame=frame,
                mode=mode,
                task_weight=task_weight,
            ),
            None,
            [*left.pose, left.velocity, *right.pose, right.velocity],
        )


def _initial_velocity(
    robot: Robot, keyframes: ReferenceTrajectory, times: FloatArray, initial: FloatArray, *, loop: bool
) -> FloatArray:
    # 从关键帧参考曲线取切线；循环接缝的两侧属于相邻周期。
    epsilon = float(np.min(np.diff(times))) * 1e-5
    duration = times[-1]
    if loop:
        left, right = (times - epsilon) % duration, (times + epsilon) % duration
        span = np.full_like(times, 2 * epsilon)
    else:
        left, right = np.maximum(0.0, times - epsilon), np.minimum(duration, times + epsilon)
        span = right - left
    a, b = keyframes.sample(robot, left), keyframes.sample(robot, right)
    velocity = np.stack([robot.difference(p, q, float(dt)) for p, q, dt in zip(a, b, span, strict=True)])
    if robot.floating:
        # mj_differentiatePos 的角速度在左采样点的轴下，转换到节点的局部轴。
        to_node = rotation_from_wxyz(initial[:, 3:7]).inv() * rotation_from_wxyz(a[:, 3:7])
        velocity[:, 3:6] = to_node.apply(velocity[:, 3:6])
    return velocity


def _add_frame_tasks(
    prepared: _PreparedTrajectory, robot: Robot, project: Project, keyframes: ReferenceTrajectory
) -> None:
    if not project.uses_frame_tasks:
        return
    settings, problem, times = project.settings, prepared.problem, prepared.times

    for name, task in project.frame_tasks.items():
        if task.mode is None or task.weight <= 0:
            continue
        frame = TaskFrame(robot, name, task.reference_body)
        if settings.frame_reference > 0:
            targets = TaskReference(frame, keyframes).sample(times)
            weight = float(np.sqrt(task.weight * settings.frame_reference / len(times)))
            for node, target in zip(prepared.nodes, targets, strict=True):
                if not node.locked:
                    problem.add_residual_block(FrameReferenceCost(frame, task.mode, target, weight), None, node.pose)
        _add_motion_costs(
            prepared,
            settings.frame_velocity,
            settings.frame_acceleration,
            frame=frame,
            mode=task.mode,
            task_weight=task.weight,
        )


def _prepare(robot: Robot, project: Project) -> _PreparedTrajectory:
    settings = project.settings
    # 非帧对齐关键帧作为独立节点加入，避免约束被舍入或漏掉。
    regular = np.linspace(0, project.duration, max(2, int(np.ceil(project.duration * settings.solve_fps)) + 1))
    protected = np.unique([0.0, project.duration, *(key.time for key in project.keyframes)])
    # 优先保留作者给出的准确时间，移除与其几乎重合的规则节点，避免极短时间步。
    extra = [time for time in regular if not np.any(np.isclose(protected, time, rtol=0, atol=1e-8))]
    times: FloatArray = np.sort(np.r_[protected, extra])
    loop = settings.terminal_condition == "loop"
    # 循环区间的两个端点必须引用不同节点。
    if loop and len(times) == 2:
        times = np.insert(times, 1, project.duration / 2)
    # 逐关节插值只初始化优化变量，不作为整段轨迹的姿态跟随目标。
    keyframes = project.keyframe_trajectory(robot)
    initial = keyframes.sample(robot, times)
    count = len(times)
    key_indices = {key.id: int(np.argmin(abs(times - key.time))) for key in project.keyframes}
    fixed = {key_indices[key.id] for key in project.keyframes if key.locked}
    for key in project.keyframes:
        if key.locked:
            initial[key_indices[key.id]] = key.qpos
    if loop:
        initial[0] = initial[-1] = keyframes.qpos[0]
        if count - 1 in fixed:
            fixed.add(0)

    # 节点持有参数数组；PyCeres 问题负责保留残差和流形对象的引用。
    problem = pyceres.Problem()
    layout = PoseLayout(robot)
    nodes: list[_SolveNode] = []
    steps = np.diff(times)
    velocities = (
        _initial_velocity(robot, keyframes, times, initial, loop=loop) if robot.model.nv else np.empty((count, 0))
    )
    quaternion_manifold = pyceres.QuaternionManifold()
    for index in range(count - int(loop)):
        pose = layout.pack(initial[index])
        if robot.joint_names:
            joints = pose[0]
            problem.add_parameter_block(joints, len(robot.joint_names))
            for joint, (lower, upper) in enumerate(zip(robot.lower, robot.upper, strict=True)):
                if np.isfinite(lower):
                    problem.set_parameter_lower_bound(joints, joint, float(lower))
                if np.isfinite(upper):
                    problem.set_parameter_upper_bound(joints, joint, float(upper))
        if robot.floating:
            problem.add_parameter_block(pose[-2], 3)
            problem.add_parameter_block(pose[-1], 4, quaternion_manifold)
        locked = index in fixed
        if locked:
            for block in pose:
                problem.set_parameter_block_constant(block)

        incoming = float(steps[index - 1]) if index > 0 or loop else None
        outgoing = float(steps[index]) if index < len(steps) else None
        mapping = KnotVelocity(robot, incoming, outgoing, stopped=not loop and index == count - 1)
        velocity = mapping.add_block(problem, initial[index], velocities[index])
        nodes.append(_SolveNode(pose, locked, mapping, velocity))
    if loop:
        # 末时刻引用首节点的姿态、速度与锁定状态。
        nodes.append(nodes[0])
    prepared = _PreparedTrajectory(times, nodes, layout, problem)
    if robot.model.nv == 0:
        return prepared

    # 姿态偏好仅来自作者显式给出的未锁定关键帧，并只作用于对应时刻。
    for key in project.keyframes:
        node = nodes[key_indices[key.id]]
        if settings.keyframe <= 0 or node.locked:
            continue
        weight = float(np.sqrt(settings.keyframe))
        target = layout.pack(np.asarray(key.qpos))
        linear = len(target) - int(robot.floating)
        for goal, block in zip(target[:linear], node.pose[:linear], strict=True):
            problem.add_residual_block(LinearReferenceCost(goal, weight), None, [block])
        if robot.floating:
            problem.add_residual_block(RotationReferenceCost(target[-1], weight), None, [node.pose[-1]])

    _add_motion_costs(prepared, settings.velocity, settings.acceleration)
    environment = robot.environment
    free = [(index, node) for index, node in enumerate(nodes) if not node.locked]
    if settings.foot_support and (
        settings.support_position > 0 or settings.support_orientation > 0 or settings.balance > 0
    ):
        data = robot.forward(initial[0])
        for index, node in free:
            robot.forward(initial[index], data)
            contacts = detect_support_contacts(foot_contacts(robot, data), nearest=True)
            problem.add_residual_block(SupportCost(robot, contacts, settings, count), None, node.pose)
    if environment.collision_pairs and (settings.self_collision > 0 or settings.ground_collision > 0):
        pairs = np.asarray(environment.collision_pairs, dtype=int)
        ground = np.any(environment.model.geom_bodyid[pairs] == 0, axis=1)
        weights = np.sqrt(np.where(ground, settings.ground_collision, settings.self_collision) / count)
        for _, node in free:
            problem.add_residual_block(
                CollisionCost(robot, weights, margin_cm=settings.collision_margin_cm),
                None,
                node.pose,
            )
    _add_frame_tasks(prepared, robot, project, keyframes)
    return prepared


def optimize(
    robot: Robot,
    project: Project,
    *,
    cancel: Event | None = None,
    progress: Callable[[int, float], None] | None = None,
) -> SolvedTrajectory:
    if cancel is not None and cancel.is_set():
        raise SolveCancelled("已取消求解")
    project = project.normalized(robot)
    prepared = _prepare(robot, project)
    if cancel is not None and cancel.is_set():
        raise SolveCancelled("已取消求解")
    if prepared.problem.num_residual_blocks() == 0:
        return prepared.result(True, "没有可优化的自由目标")
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
    cost = float(summary.final_cost)
    converged = summary.termination_type in (pyceres.TerminationType.CONVERGENCE, pyceres.TerminationType.USER_SUCCESS)
    message = f"迭代 {callback.iterations} · 代价 {cost:.5g}"
    if not converged:
        message = f"达到最大迭代次数 · {message}"
    return prepared.result(converged, message)
