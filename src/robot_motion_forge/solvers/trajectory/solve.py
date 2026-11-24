"""PyCeres 整段轨迹优化，关键帧与关节限位为硬约束。"""

from collections.abc import Callable
from dataclasses import dataclass
from threading import Event

import numpy as np
import pyceres

from robot_motion_forge.motion.project import Project
from robot_motion_forge.motion.trajectory import ReferenceTrajectory, SolvedTrajectory, validate_trajectory
from robot_motion_forge.robot.closure import ClosureError
from robot_motion_forge.robot.definition import ControlMode
from robot_motion_forge.robot.kinematics import ControlFrameKinematics
from robot_motion_forge.robot.model import Robot
from robot_motion_forge.utils.array_types import FloatArray

from .costs.collision import CollisionCost, collision_model_for
from .costs.reference import FrameReferenceCost, LinearReferenceCost, RotationReferenceCost, sample_frame_reference
from .costs.smoothness import SmoothnessCost
from .costs.support import SupportCost, initial_support_contacts
from .parameters import Jacobians, KnotVelocityMap, Parameters, PoseLayout


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
    velocity_mapping: KnotVelocityMap
    velocity: FloatArray

    @property
    def fixed(self) -> bool:
        return self.locked and self.velocity_mapping.stopped


@dataclass(frozen=True)
class _PreparedTrajectory:
    times: FloatArray
    nodes: list[_SolveNode]
    layout: PoseLayout
    problem: pyceres.Problem

    def result(self, converged: bool, message: str) -> SolvedTrajectory:
        robot = self.layout.robot
        qpos = np.empty((len(self.nodes), robot.model.nq))
        qvel = np.empty((len(self.nodes), robot.model.nv))
        for index, node in enumerate(self.nodes):
            pose, lift = self.layout.reconstruct(node.pose, jacobian=True)
            velocity, _, _ = node.velocity_mapping.decode(pose, node.velocity)
            qpos[index] = pose
            qvel[index] = velocity if lift is None else lift @ velocity
        if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
            raise ValueError("求解器返回了非有限轨迹")
        trajectory = SolvedTrajectory(self.times, qpos, qvel, converged, message)
        if robot.closure is not None:
            validate_trajectory(robot, float(self.times[-1]), trajectory)
        return trajectory


class _ClosureFeasibilityCost(pyceres.CostFunction):
    """没有几何代价时也拒绝无法闭合的候选姿态。"""

    def __init__(self, layout: PoseLayout) -> None:
        super().__init__()
        self.layout = layout
        self.set_parameter_block_sizes(layout.sizes)
        self.set_num_residuals(1)

    def Evaluate(self, parameters: Parameters, residuals: FloatArray, jacobians: Jacobians) -> bool:
        try:
            self.layout.reconstruct(parameters)
        except ClosureError:
            return False
        residuals.fill(0.0)
        if jacobians is not None:
            for block in jacobians:
                if block is not None:
                    block.fill(0.0)
        return True


def _add_smoothness_costs(
    prepared: _PreparedTrajectory,
    velocity: float,
    acceleration: float,
    *,
    frame: ControlFrameKinematics | None = None,
    mode: ControlMode = "pose",
    task_weight: float = 1.0,
) -> None:
    if velocity == 0 and acceleration == 0:
        return
    duration = float(prepared.times[-1])
    for index, step in enumerate(np.diff(prepared.times)):
        left, right = prepared.nodes[index : index + 2]
        # 两端的姿态和速度均已固定时，该段平滑代价为常量。
        if left.fixed and right.fixed:
            continue
        prepared.problem.add_residual_block(
            SmoothnessCost(
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


def _add_frame_tasks(prepared: _PreparedTrajectory, project: Project, keyframes: ReferenceTrajectory) -> bool:
    settings, problem, times = project.settings, prepared.problem, prepared.times
    if max(settings.frame_reference, settings.frame_velocity, settings.frame_acceleration) <= 0:
        return False

    enabled = False
    for name, task in project.frame_tasks.items():
        if task.mode is None or task.weight <= 0:
            continue
        enabled = True
        frame = ControlFrameKinematics(prepared.layout.robot, name, task.reference_body)
        if settings.frame_reference > 0:
            targets = sample_frame_reference(frame, keyframes, times, task.mode)
            weight = float(np.sqrt(task.weight * settings.frame_reference / len(times)))
            for node, target in zip(prepared.nodes, targets, strict=True):
                if not node.locked:
                    problem.add_residual_block(
                        FrameReferenceCost(prepared.layout, frame, target, weight), None, node.pose
                    )
        _add_smoothness_costs(
            prepared,
            settings.frame_velocity,
            settings.frame_acceleration,
            frame=frame,
            mode=task.mode,
            task_weight=task.weight,
        )
    return enabled


def _solve_node_times(project: Project) -> FloatArray:
    # 每个关键帧都作为分段边界，保留作者给出的准确时间。
    boundaries = np.asarray(sorted(key.time for key in project.keyframes), dtype=float)
    # 沿用规则网格插入关键帧后的节点预算，几乎重合的规则节点只计一次。
    regular_count = max(2, int(np.ceil(project.duration * project.settings.solve_fps)) + 1)
    regular = np.linspace(0.0, project.duration, regular_count)
    right = np.searchsorted(boundaries, regular).clip(1, len(boundaries) - 1)
    distance = np.minimum(regular - boundaries[right - 1], boundaries[right] - regular)
    interval_count = len(boundaries) - 1 + int(np.count_nonzero(distance > 1e-8))

    # 各段至少有一个区间，按与时长比例的差额补足或扣减整数区间。
    ideal = np.diff(boundaries) / project.duration * interval_count
    intervals = np.maximum(1, np.floor(ideal).astype(int))
    remaining = interval_count - int(intervals.sum())
    while remaining > 0:
        intervals[int(np.argmax(ideal - intervals))] += 1
        remaining -= 1
    while remaining < 0:
        intervals[int(np.argmax(np.where(intervals > 1, intervals - ideal, -np.inf)))] -= 1
        remaining += 1

    parts: list[FloatArray] = [boundaries[:1]]
    for start, end, count in zip(boundaries[:-1], boundaries[1:], intervals, strict=True):
        parts.append(np.linspace(float(start), float(end), int(count) + 1)[1:])
    return np.concatenate(parts)


def _build_nodes(
    robot: Robot, project: Project, keyframes: ReferenceTrajectory, times: FloatArray, key_indices: dict[str, int]
) -> _PreparedTrajectory:
    loop = project.settings.terminal_condition != "rest"
    translation_loop = keyframes.terminal_condition == "translation_loop"
    # 独立坐标插值只初始化优化变量，不作为整段轨迹的姿态跟随目标。
    initial, velocities = keyframes.sample_initial_state(robot, times)
    count = len(times)
    fixed = {key_indices[key.id] for key in project.keyframes if key.locked}
    stops = {key_indices[key.id] for key in project.keyframes if key.zero_velocity}
    qpos_indices = np.r_[:7, robot.qadr] if robot.floating else robot.qadr
    for key in project.keyframes:
        if key.locked:
            initial[key_indices[key.id], qpos_indices] = np.asarray(key.qpos)[qpos_indices]
    if loop:
        shared = qpos_indices[2:] if translation_loop else qpos_indices
        initial[np.ix_([0, count - 1], shared)] = keyframes.qpos[0, shared]
        if translation_loop:
            # 两端从各自的水平曲线取速度，以两者的均值初始化共享速度。
            velocities[0, :2] = (velocities[0, :2] + velocities[-1, :2]) / 2
        if not translation_loop and count - 1 in fixed:
            fixed.add(0)
        if count - 1 in stops:
            stops.add(0)
    else:
        stops.update((0, count - 1))

    # 节点持有参数数组；PyCeres 问题负责保留残差和流形对象的引用。
    problem = pyceres.Problem()
    layout = PoseLayout(robot)
    nodes: list[_SolveNode] = []
    steps = np.diff(times)
    for index in range(count - int(loop)):
        locked = index in fixed
        pose = layout.add_blocks(problem, initial[index], locked=locked)

        incoming = float(steps[index - 1]) if index > 0 or loop else None
        outgoing = float(steps[index]) if index < len(steps) else None
        mapping = KnotVelocityMap(robot, incoming, outgoing, stopped=index in stops)
        velocity = mapping.add_block(problem, initial[index], velocities[index])
        nodes.append(_SolveNode(pose, locked, mapping, velocity))
    if loop:
        first = nodes[0]
        if translation_loop:
            locked = count - 1 in fixed
            pose = layout.add_translated_endpoint(problem, first.pose, initial[-1, :2], locked=locked)
            nodes.append(_SolveNode(pose, locked, first.velocity_mapping, first.velocity))
        else:
            # 原地循环的末时刻引用首节点的姿态、速度与锁定状态。
            nodes.append(first)
    return _PreparedTrajectory(times, nodes, layout, problem)


def _add_costs(
    prepared: _PreparedTrajectory, project: Project, keyframes: ReferenceTrajectory, key_indices: dict[str, int]
) -> None:
    layout, nodes, problem = prepared.layout, prepared.nodes, prepared.problem
    settings, robot = project.settings, layout.robot
    count = len(nodes)

    # 姿态偏好仅来自作者显式给出的未锁定关键帧，并只作用于对应时刻。
    for key in project.keyframes:
        node = nodes[key_indices[key.id]]
        if settings.keyframe <= 0 or node.locked:
            continue
        weight = float(np.sqrt(settings.keyframe))
        target = layout.pack(np.asarray(key.qpos))
        linear = len(target) - int(robot.floating)
        # 平移循环的另一端可以锁定共享参数块，本端只为仍可调整的量添加偏好。
        for goal, block in zip(target[:linear], node.pose[:linear], strict=True):
            if not problem.is_parameter_block_constant(block):
                problem.add_residual_block(LinearReferenceCost(goal, weight), None, [block])
        if robot.floating and not problem.is_parameter_block_constant(node.pose[-1]):
            problem.add_residual_block(RotationReferenceCost(target[-1], weight), None, [node.pose[-1]])

    _add_smoothness_costs(prepared, settings.velocity, settings.acceleration)
    free = [node for node in nodes if not node.locked]
    has_geometric_costs = False
    if settings.foot_support and (
        settings.support_position > 0 or settings.support_orientation > 0 or settings.balance > 0
    ):
        data = None
        for node in free:
            data = robot.forward(layout.reconstruct(node.pose)[0], data)
            contacts = initial_support_contacts(robot, data)
            problem.add_residual_block(SupportCost(layout, contacts, settings, count), None, node.pose)
        has_geometric_costs = bool(free)
    if free and (settings.self_collision > 0 or settings.ground_collision > 0):
        collision = collision_model_for(robot)
        if collision.collision_pairs.size:
            has_geometric_costs = True
            weights = np.sqrt(
                np.where(collision.ground_collision_mask, settings.ground_collision, settings.self_collision) / count
            )
            for node in free:
                problem.add_residual_block(
                    CollisionCost(layout, collision, weights, margin_cm=settings.collision_margin_cm),
                    None,
                    node.pose,
                )
    has_frame_costs = _add_frame_tasks(prepared, project, keyframes)
    if robot.closure is not None and not (has_geometric_costs or has_frame_costs) and problem.num_residual_blocks():
        feasibility = _ClosureFeasibilityCost(layout)
        for node in free:
            problem.add_residual_block(feasibility, None, node.pose)


def _prepare(robot: Robot, project: Project) -> _PreparedTrajectory:
    times = _solve_node_times(project)
    # 区间两端不能在同一个代价中重复引用循环共享的参数块。
    if project.settings.terminal_condition != "rest" and len(times) == 2:
        times = np.insert(times, 1, project.duration / 2)
    keyframes = project.keyframe_trajectory(robot)
    key_indices = {key.id: int(np.argmin(abs(times - key.time))) for key in project.keyframes}
    prepared = _build_nodes(robot, project, keyframes, times, key_indices)
    # 全部节点的姿态和速度均已固定时，直接生成轨迹。
    if len(robot.independent_dofs) and not all(node.fixed for node in prepared.nodes):
        _add_costs(prepared, project, keyframes, key_indices)
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
