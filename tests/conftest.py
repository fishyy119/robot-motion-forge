"""测试模型临时生成，不在仓库中维护机器人资产。"""

from pathlib import Path

import pytest

from robot_motion_forge.motion.project import Project
from robot_motion_forge.robot.definition import ControlFrame, HomePose, ModelConfig, RobotDefinition
from robot_motion_forge.robot.model import Robot


@pytest.fixture
def robot(tmp_path: Path) -> Robot:
    model_path = tmp_path / "robot.xml"
    model_path.write_text(
        """<mujoco>
      <compiler angle="radian"/>
      <worldbody>
        <body name="base" pos="0 0 .6">
          <freejoint/>
          <geom name="trunk" type="box" size=".08 .08 .08" mass="2"/>
          <body name="left_foot" pos="0 .16 -.55">
            <geom name="left_sole" type="sphere" size=".05" mass=".1"/>
          </body>
          <body name="right_foot" pos="0 -.16 -.55">
            <geom name="right_sole" type="sphere" size=".05" mass=".1"/>
          </body>
          <body name="arm">
            <joint name="shoulder" axis="0 0 1" range="-1.5 1.5"/>
            <geom type="capsule" fromto="0 0 0 .4 0 0" size=".02" mass=".1"/>
            <body name="hand" pos=".4 0 0">
              <geom type="sphere" size=".025" mass=".1"/>
            </body>
          </body>
        </body>
      </worldbody>
      <actuator><motor joint="shoulder" ctrlrange="-2 2"/></actuator>
    </mujoco>""",
        encoding="utf-8",
    )
    config = RobotDefinition(
        robot=ModelConfig(name="test", repository_root=str(tmp_path), model="robot.xml"),
        home=HomePose(joints_rad={"shoulder": 0.0}, quaternion_wxyz=(1.0, 0.0, 0.0, 0.0)),
        frames={
            "torso": ControlFrame(body="base", fixed_mode="pose"),
            "left_foot": ControlFrame(body="left_foot", offset_m=(0.0, 0.0, -0.05), control_mode="pose"),
            "right_foot": ControlFrame(body="right_foot", offset_m=(0.0, 0.0, -0.05), control_mode="pose"),
            "hand": ControlFrame(body="hand"),
        },
    )
    return Robot(config, config_path=tmp_path / "configs" / "robots" / "robot.json")


@pytest.fixture
def project(robot: Robot) -> Project:
    project = Project.new(robot)
    project.duration = 0.6
    project.keyframes[-1].time = project.duration
    project.settings.solve_fps = 10
    return project
