"""各页签的控件构建、事件绑定与局部显示状态。"""

from .arrangement import ArrangementActions, ArrangementTab
from .display import DisplayTab
from .pose import PoseActions, PoseTab
from .project import ProjectActions, ProjectTab
from .solve import SolveActions, SolveTab
from .timeline import TimelineTab

__all__ = [
    "ArrangementActions",
    "ArrangementTab",
    "DisplayTab",
    "PoseActions",
    "PoseTab",
    "ProjectActions",
    "ProjectTab",
    "SolveActions",
    "SolveTab",
    "TimelineTab",
]
