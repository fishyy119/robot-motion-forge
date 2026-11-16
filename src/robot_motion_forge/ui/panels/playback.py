"""底部浮动面板布局：组装时间轴页与显示页。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..tabs import DisplayTab, TimelineTab

if TYPE_CHECKING:
    from robot_motion_forge.motion.project import Project

    from ..controls import Controls


class PlaybackPanel:
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
        panel.float(x=16, y=-16, width=1200)
