"""求解页：动作设置、优化参数与末端轨迹配置。"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Protocol

from robot_motion_forge.motion.types import EndEffectorTask

if TYPE_CHECKING:
    from viser import GuiDropdownHandle, GuiSliderHandle

    from robot_motion_forge.motion.project import Project
    from robot_motion_forge.motion.types import EndEffectorTaskMode
    from robot_motion_forge.robot.model import Robot

    from ..controls import Controls

_SUPPORT_FIELDS = ("support_position", "support_orientation", "balance", "support_margin_cm")
_END_EFFECTOR_FIELDS = ("end_effector_reference", "end_effector_velocity", "end_effector_acceleration")
_TASK_MODES: dict[str, EndEffectorTaskMode] = {
    "禁用": "none",
    "位置": "position",
    "朝向": "orientation",
    "位姿": "pose",
}


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
    def set_duration(self, value: float) -> None: ...
    def set_fps(self, value: float) -> None: ...
    def set_foot_support(self, value: bool) -> None: ...
    def set_end_effector_mode(self, name: str, value: EndEffectorTaskMode) -> None: ...
    def set_end_effector_reference(self, name: str, value: str | None) -> None: ...
    def set_end_effector_weight(self, name: str, value: float) -> None: ...
    def set_setting(self, name: str, value: float | int) -> None: ...


class SolveTab:
    def __init__(self, controls: Controls, *, robot: Robot, project: Project) -> None:
        gui = controls.gui
        self._controls = controls
        tasks_enabled = any(task.mode != "none" for task in project.end_effector_tasks.values())
        with gui.add_folder("动作设置"):
            self._duration = gui.add_number("动作时长 / s", initial_value=project.duration, step=0.1)
            self._fps = gui.add_number("导出 FPS", initial_value=project.fps, step=1.0)
        with gui.add_folder("轨迹优化"):
            self._solve_button = controls.button("优化整段轨迹", primary=True)
            self._cancel_button = controls.button("取消求解")
            self._foot_support = gui.add_checkbox("考虑足部支撑", initial_value=project.settings.foot_support)
            self._setting_inputs: dict[str, GuiSliderHandle[float | int]] = {}
            # 拖动步长与小数输入精度分开设置，保留手动输入的精确值。
            settings: dict[str, _SliderConfig] = {
                "style": _SliderConfig(label="动作风格", min=0.0, max=10.0, step=0.1, precision=3),
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
                "end_effector_reference": _SliderConfig(label="末端轨迹跟随", min=0.0, max=10.0, step=0.1, precision=3),
                "end_effector_velocity": _SliderConfig(label="末端速度正则", min=0.0, max=0.1, step=0.001, precision=4),
                "end_effector_acceleration": _SliderConfig(
                    label="末端加速度平滑", min=0.0, max=0.1, step=0.001, precision=4
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
                    and (field not in _END_EFFECTOR_FIELDS or tasks_enabled),
                )
                handle.precision = max(handle.precision, config.precision)
                self._setting_inputs[field] = handle
        self._reference_bodies: dict[str, str | None] = {"世界": None}
        self._reference_bodies.update(
            {f"连杆 · {name}": name for body in robot.body_ids if (name := robot.model.body(body).name)}
        )
        self._task_inputs: dict[str, _TaskInputs] = {}
        with gui.add_folder("末端轨迹", expand_by_default=False):
            for name, effector in robot.end_effectors.items():
                task = project.end_effector_tasks.get(name, EndEffectorTask())
                with gui.add_folder(name, expand_by_default=False):
                    mode = gui.add_dropdown(
                        "模式",
                        options=tuple(_TASK_MODES),
                        initial_value=next(label for label, value in _TASK_MODES.items() if value == task.mode),
                    )
                    reference = gui.add_dropdown(
                        "参考坐标系",
                        options=tuple(label for label, body in self._reference_bodies.items() if body != effector.body),
                        initial_value=next(
                            label for label, body in self._reference_bodies.items() if body == task.reference_body
                        ),
                        visible=task.mode != "none",
                    )
                    weight = gui.add_slider(
                        "相对权重",
                        min=0.0,
                        max=max(10.0, task.weight),
                        step=0.1,
                        initial_value=task.weight,
                        marks=(),
                        visible=task.mode != "none",
                    )
                    weight.precision = 3
                    self._task_inputs[name] = _TaskInputs(mode, reference, weight)

    def bind_actions(self, actions: SolveActions) -> None:
        controls = self._controls
        controls.input(self._duration, actions.set_duration)
        controls.input(self._fps, actions.set_fps)
        controls.input(self._foot_support, actions.set_foot_support)
        controls.click(self._solve_button, actions.start)
        controls.click(self._cancel_button, actions.cancel)
        for field, handle in self._setting_inputs.items():
            controls.input(handle, lambda value, field=field: actions.set_setting(field, value))
        for name, inputs in self._task_inputs.items():
            controls.input(
                inputs.mode, lambda value, name=name: actions.set_end_effector_mode(name, _TASK_MODES[value])
            )
            controls.input(
                inputs.reference,
                lambda value, name=name: actions.set_end_effector_reference(name, self._reference_bodies[value]),
            )
            controls.input(inputs.weight, lambda value, name=name: actions.set_end_effector_weight(name, value))

    def refresh(self, project: Project) -> None:
        self._duration.value, self._fps.value = project.duration, project.fps
        self._foot_support.value = project.settings.foot_support
        tasks_enabled = any(task.mode != "none" for task in project.end_effector_tasks.values())
        for name, handle in self._setting_inputs.items():
            value: float | int = getattr(project.settings, name)
            # 载入工程或撤销时保留已有参数，不因滑块的常用范围截断数值。
            self._controls.sync_slider(handle, value)
            if name in _SUPPORT_FIELDS:
                handle.visible = project.settings.foot_support
            elif name in _END_EFFECTOR_FIELDS:
                handle.visible = tasks_enabled
        for name, inputs in self._task_inputs.items():
            task = project.end_effector_tasks.get(name, EndEffectorTask())
            inputs.mode.value = next(label for label, value in _TASK_MODES.items() if value == task.mode)
            inputs.reference.value = next(
                label for label, body in self._reference_bodies.items() if body == task.reference_body
            )
            self._controls.sync_slider(inputs.weight, task.weight)
            inputs.reference.visible = inputs.weight.visible = task.mode != "none"

    def set_busy(self, busy: bool) -> None:
        self._solve_button.disabled = busy
        self._cancel_button.disabled = not busy
