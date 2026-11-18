"""求解页：动作设置、轨迹优化、坐标系轨迹配置与动作导出。"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, NamedTuple, Protocol

from robot_motion_forge.motion.types import FrameTask, TerminalCondition
from robot_motion_forge.robot.definition import ControlMode

from ..controls import CONTROL_MODES

if TYPE_CHECKING:
    from viser import ClientHandle, GuiButtonHandle, GuiDropdownHandle, GuiSliderHandle

    from robot_motion_forge.motion.project import Project
    from robot_motion_forge.robot.model import Robot

    from ..controls import Controls

_SUPPORT_FIELDS = ("support_position", "support_orientation", "balance", "support_margin_cm")
_FRAME_FIELDS = ("frame_reference", "frame_velocity", "frame_acceleration")
_TASK_MODES: dict[str, ControlMode | None] = {"禁用": None, **CONTROL_MODES}
_TERMINAL_LABELS: dict[TerminalCondition, str] = {"rest": "静止", "loop": "循环"}


class _SliderConfig(NamedTuple):
    label: str
    min: float | int
    max: float | int
    step: float | int
    precision: int


class _TaskInputs(NamedTuple):
    mode: GuiDropdownHandle[str]
    reference: GuiDropdownHandle[str]
    weight: GuiSliderHandle[float]


class SolveActions(Protocol):
    def start(self) -> None: ...
    def cancel(self) -> None: ...
    def export(self, client: ClientHandle) -> None: ...
    def set_timing(self, *, duration: float | None = None, fps: float | None = None) -> None: ...
    def set_frame_task(self, name: str, task: FrameTask) -> None: ...
    def set_setting(self, name: str, value: float | int | str) -> None: ...


class SolveTab:
    def __init__(self, controls: Controls, *, robot: Robot, project: Project) -> None:
        gui = controls.gui
        self._controls = controls
        tasks_enabled = any(task.mode is not None for task in project.frame_tasks.values())
        with gui.add_folder("动作设置"):
            self._duration = gui.add_number("动作时长 / s", initial_value=project.duration, step=0.1)
            self._fps = gui.add_number("导出 FPS", initial_value=project.fps, step=1.0)
        with gui.add_folder("轨迹优化"):
            self._solve_button = controls.button("优化整段轨迹", primary=True)
            self._cancel_button = controls.button("取消求解")
            self._export_button = controls.button("导出求解 PKL", disabled=True)
            self._terminal_condition = gui.add_dropdown(
                "终端条件",
                options=tuple(_TERMINAL_LABELS.values()),
                initial_value=_TERMINAL_LABELS[project.settings.terminal_condition],
            )
            self._foot_support = gui.add_checkbox("考虑足部支撑", initial_value=project.settings.foot_support)
            self._setting_inputs: dict[str, GuiSliderHandle[float | int]] = {}
            # 拖动步长与小数输入精度分开设置，保留手动输入的精确值。
            settings: dict[str, _SliderConfig] = {
                "velocity": _SliderConfig(label="速度正则", min=0.0, max=0.1, step=0.001, precision=4),
                "acceleration": _SliderConfig(label="加速度平滑", min=0.0, max=0.1, step=0.001, precision=4),
                "keyframe": _SliderConfig(label="未锁定帧权重", min=0.0, max=1000.0, step=10.0, precision=3),
                "self_collision": _SliderConfig(label="自碰撞权重", min=0.0, max=10.0, step=0.1, precision=3),
                "ground_collision": _SliderConfig(label="地面碰撞权重", min=0.0, max=10.0, step=0.1, precision=3),
                "collision_margin_cm": _SliderConfig(label="自碰撞间隙 / cm", min=0.0, max=5.0, step=0.1, precision=2),
                "support_position": _SliderConfig(label="足底贴地权重", min=0.0, max=10.0, step=0.1, precision=3),
                "support_orientation": _SliderConfig(label="足底水平权重", min=0.0, max=10.0, step=0.1, precision=3),
                "balance": _SliderConfig(label="重心支撑权重", min=0.0, max=10.0, step=0.1, precision=3),
                "support_margin_cm": _SliderConfig(label="支撑裕量 / cm", min=0.0, max=5.0, step=0.1, precision=2),
                "frame_reference": _SliderConfig(label="坐标系轨迹跟随", min=0.0, max=10.0, step=0.1, precision=3),
                "frame_velocity": _SliderConfig(label="坐标系速度正则", min=0.0, max=0.1, step=0.001, precision=4),
                "frame_acceleration": _SliderConfig(
                    label="坐标系加速度平滑", min=0.0, max=0.1, step=0.001, precision=4
                ),
                "solve_fps": _SliderConfig(label="求解采样 FPS", min=1.0, max=30.0, step=1.0, precision=3),
                "max_iterations": _SliderConfig(label="最大迭代次数", min=1, max=500, step=1, precision=0),
            }
            for field, config in settings.items():
                value: float | int = getattr(project.settings, field)
                handle = gui.add_slider(
                    config.label,
                    min=min(config.min, value),
                    max=max(config.max, value),
                    step=config.step,
                    initial_value=value,
                    marks=(),
                    visible=(field not in _SUPPORT_FIELDS or project.settings.foot_support)
                    and (field not in _FRAME_FIELDS or tasks_enabled),
                )
                handle.precision = max(handle.precision, config.precision)
                self._setting_inputs[field] = handle
        self._reference_labels: dict[str | None, str] = {None: "世界"}
        for body_id in robot.body_ids:
            if body_name := robot.model.body(body_id).name:
                self._reference_labels[body_name] = "基座" if body_id == robot.root_id else f"连杆 · {body_name}"
        self._task_defaults = {
            name: FrameTask(reference_body=frame.trajectory_reference_body) for name, frame in robot.frames.items()
        }
        self._task_inputs: dict[str, _TaskInputs] = {}
        with gui.add_folder("坐标系轨迹", expand_by_default=False):
            for name, frame in robot.frames.items():
                task = project.frame_tasks.get(name, self._task_defaults[name])
                with gui.add_folder(name, expand_by_default=False):
                    mode = gui.add_dropdown(
                        "模式",
                        options=tuple(_TASK_MODES),
                        initial_value=next(label for label, value in _TASK_MODES.items() if value == task.mode),
                    )
                    reference = gui.add_dropdown(
                        "参考坐标系",
                        options=tuple(label for body, label in self._reference_labels.items() if body != frame.body),
                        initial_value=self._reference_labels[task.reference_body],
                        visible=task.mode is not None,
                    )
                    weight = gui.add_slider(
                        "相对权重",
                        min=0.0,
                        max=max(10.0, task.weight),
                        step=0.1,
                        initial_value=task.weight,
                        marks=(),
                        visible=task.mode is not None,
                    )
                    weight.precision = 3
                    self._task_inputs[name] = _TaskInputs(mode, reference, weight)

    def bind_actions(self, actions: SolveActions) -> None:
        controls = self._controls
        controls.input(self._duration, lambda value: actions.set_timing(duration=value))
        controls.input(self._fps, lambda value: actions.set_timing(fps=value))
        controls.input(
            self._terminal_condition,
            lambda label: actions.set_setting(
                "terminal_condition", next(condition for condition, text in _TERMINAL_LABELS.items() if text == label)
            ),
        )
        controls.input(self._foot_support, lambda value: actions.set_setting("foot_support", value))
        controls.click(self._solve_button, actions.start)
        controls.click(self._cancel_button, actions.cancel)

        def prepare_export(client: ClientHandle, _target: GuiButtonHandle) -> Callable[[], None]:
            return partial(actions.export, client)

        # 下载只发送给触发导出的客户端，生成数据仍统一在主线程执行。
        self._export_button.on_click(controls.client_action(prepare_export))

        for field, handle in self._setting_inputs.items():
            controls.input(handle, lambda value, field=field: actions.set_setting(field, value))

        reference_bodies = {label: body for body, label in self._reference_labels.items()}

        def prepare_task(name: str, _client: ClientHandle, _target: object) -> Callable[[], None]:
            inputs = self._task_inputs[name]
            # 入队前保存同一任务的全部字段；编辑器统一提交并处理撤销合并。
            task = FrameTask(
                mode=_TASK_MODES[inputs.mode.value],
                reference_body=reference_bodies[inputs.reference.value],
                weight=inputs.weight.value,
            )
            return partial(actions.set_frame_task, name, task)

        for name, inputs in self._task_inputs.items():
            for handle in inputs:
                handle.on_update(controls.client_action(partial(prepare_task, name)))

    def refresh(self, project: Project) -> None:
        self._duration.value, self._fps.value = project.duration, project.fps
        self._terminal_condition.value = _TERMINAL_LABELS[project.settings.terminal_condition]
        self._foot_support.value = project.settings.foot_support
        tasks_enabled = any(task.mode is not None for task in project.frame_tasks.values())
        for name, handle in self._setting_inputs.items():
            value: float | int = getattr(project.settings, name)
            # 载入工程或撤销时保留已有参数，不因滑块的常用范围截断数值。
            self._controls.sync_slider(handle, value)
            if name in _SUPPORT_FIELDS:
                handle.visible = project.settings.foot_support
            elif name in _FRAME_FIELDS:
                handle.visible = tasks_enabled
        for name, inputs in self._task_inputs.items():
            task = project.frame_tasks.get(name, self._task_defaults[name])
            inputs.mode.value = next(label for label, value in _TASK_MODES.items() if value == task.mode)
            inputs.reference.value = self._reference_labels[task.reference_body]
            self._controls.sync_slider(inputs.weight, task.weight)
            inputs.reference.visible = inputs.weight.visible = task.mode is not None

    def set_actions(self, *, busy: bool, solved: bool) -> None:
        self._solve_button.disabled = busy
        self._cancel_button.disabled = not busy
        self._export_button.disabled = not solved
