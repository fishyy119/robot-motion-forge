"""侧栏布局：组装编排、姿态、求解与工程页。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..tabs import ArrangementTab, PoseTab, ProjectTab, SolveTab
from ..theme import THEME_STYLE

if TYPE_CHECKING:
    from pathlib import Path

    from robot_motion_forge.motion.project import Project
    from robot_motion_forge.robot.model import FloatArray, Robot

    from ..controls import Controls


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
        gui.add_html(THEME_STYLE)
        tabs = gui.add_tab_group()
        with tabs.add_tab("编排"):
            self.arrangement = ArrangementTab(controls)
        with tabs.add_tab("姿态"):
            self.pose = PoseTab(controls, robot=robot, qpos=qpos)
        with tabs.add_tab("求解"):
            self.solve = SolveTab(controls, robot=robot, project=project)
        with tabs.add_tab("工程"):
            self.project = ProjectTab(controls, project_path=project_path)
