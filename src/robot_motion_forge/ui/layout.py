"""编辑器侧栏与底部浮动面板布局。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .tabs.arrangement import ArrangementTab
from .tabs.display import DisplayTab
from .tabs.joint_curves import JointCurvesTab
from .tabs.pose import PoseTab
from .tabs.project import ProjectTab
from .tabs.solve import SolveTab
from .tabs.timeline import TimelineTab

if TYPE_CHECKING:
    from pathlib import Path

    from robot_motion_forge.motion.project import Project
    from robot_motion_forge.robot.model import Robot
    from robot_motion_forge.utils.array_types import FloatArray

    from .controls import Controls


class Sidebar:
    def __init__(
        self,
        controls: Controls,
        *,
        robot: Robot,
        project: Project,
        qpos: FloatArray,
        project_path: Path,
    ) -> None:
        gui = controls.gui
        gui.set_panel_label(f"Robot Motion Forge · {robot.config.robot.name}")
        gui.main_panel.dock_right()
        gui.main_panel.set_width(440)
        gui.main_panel.expand()
        tabs = gui.add_tab_group()
        with tabs.add_tab("编排"):
            self.arrangement = ArrangementTab(controls, duration=project.duration)
        with tabs.add_tab("姿态"):
            self.pose = PoseTab(controls, robot=robot, qpos=qpos)
        with tabs.add_tab("求解"):
            self.solve = SolveTab(controls, robot=robot, project=project)
        with tabs.add_tab("关节曲线"):
            self.joint_curves = JointCurvesTab(controls, robot=robot)
        with tabs.add_tab("工程"):
            self.project = ProjectTab(controls, project_root=robot.project_root, project_path=project_path)


class BottomPanel:
    def __init__(
        self,
        controls: Controls,
        *,
        project: Project,
        cursor: float,
    ) -> None:
        panel = controls.gui.add_panel()
        self.timeline = TimelineTab(controls, tab=panel.add_tab("时间轴"), project=project, cursor=cursor)
        with panel.add_tab("显示"):
            self.display = DisplayTab(controls)
        panel.float(x=16, y=-16)
