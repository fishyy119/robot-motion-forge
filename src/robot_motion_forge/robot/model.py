"""MuJoCo 机器人适配层：运动学、流形运算与限位。没有 UI 依赖。"""

import os
from pathlib import Path
from typing import cast

import json5
import mujoco as mj
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation

from .definition import EndEffector, RobotDefinition
from .environment import Environment

type FloatArray = NDArray[np.float64]


def rotation(wxyz: ArrayLike) -> Rotation:
    return Rotation.from_quat(np.asarray(wxyz)[..., [1, 2, 3, 0]])


def quaternion(rot: Rotation) -> FloatArray:
    return rot.as_quat()[..., [3, 0, 1, 2]]


class Robot:
    def __init__(self, config: RobotDefinition, *, config_path: Path) -> None:
        self.config = config
        self.config_path = config_path.expanduser().resolve()
        repository_root = Path(os.path.expandvars(config.robot.repository_root)).expanduser()
        self.repository_root = (self.config_path.parent / repository_root).resolve()
        if not self.repository_root.is_dir():
            raise ValueError(f"找不到机器人仓库目录：{self.repository_root}")
        model_path = Path(os.path.expandvars(config.robot.model)).expanduser()
        self.model_path = (self.config_path.parent / model_path).resolve()
        if not self.model_path.is_file():
            raise ValueError(f"找不到本地模型：{self.model_path}")
        spec = mj.MjSpec.from_file(str(self.model_path))
        self.end_effector_sites = {name: f"rmf/end_effectors/{name}" for name in config.end_effectors}
        for name, end_effector in config.end_effectors.items():
            body = cast(mj.MjsBody | None, spec.body(end_effector.body))
            if body is None:
                raise ValueError(f"找不到机器人连杆：{end_effector.body}")
            # Mink 使用真实末端原点的 site 雅可比，足底偏移也参与位置任务。
            body.add_site(name=self.end_effector_sites[name], pos=end_effector.offset, rgba=(0, 0, 0, 0))
        self.model: mj.MjModel = spec.compile()
        m = self.model
        self.root_id = self.body_id(config.robot.root_body)
        roots = np.flatnonzero(m.body_parentid[1:] == 0) + 1
        if len(roots) != 1 or roots[0] != self.root_id:
            raise ValueError("root_body 必须指向模型唯一的顶层根连杆；请加载纯机器人 MJCF")
        if any(np.any(ids == 0) for ids in (m.geom_bodyid, m.site_bodyid, m.cam_bodyid, m.light_bodyid)):
            raise ValueError("模型包含 world 上的几何、site、相机或灯光；请加载纯机器人 MJCF，地面由项目提供")
        self.body_ids = list(range(1, m.nbody))
        free = np.flatnonzero(m.jnt_type == mj.mjtJoint.mjJNT_FREE)
        self.floating = len(free) == 1
        if len(free) > 1 or (self.floating and (m.jnt_qposadr[free[0]] != 0 or m.jnt_bodyid[free[0]] != self.root_id)):
            raise ValueError("当前适配器要求根部为唯一自由关节，或使用固定基座模型")
        if np.any(m.jnt_type == mj.mjtJoint.mjJNT_BALL):
            raise ValueError("当前适配器支持自由基座与 hinge/slide 关节；球关节需新增适配器")
        if m.neq:
            raise ValueError("当前适配器支持开链机器人；闭链 / equality 约束需要专用适配器")
        if m.opt.disableflags & mj.mjtDisableBit.mjDSBL_CONTACT:
            raise ValueError("模型关闭了接触，无法进行碰撞与接触验证")
        self.joint_ids = [i for i in range(m.njnt) if m.jnt_type[i] != mj.mjtJoint.mjJNT_FREE]
        self.joint_names: list[str] = [m.joint(i).name for i in self.joint_ids]
        if not all(self.joint_names):
            raise ValueError("可编辑关节必须有名称")
        self.qadr = np.asarray([m.jnt_qposadr[i] for i in self.joint_ids], dtype=int)
        self.vadr = np.asarray([m.jnt_dofadr[i] for i in self.joint_ids], dtype=int)
        self.lower: FloatArray = np.asarray(
            [m.jnt_range[i, 0] if m.jnt_limited[i] else -np.inf for i in self.joint_ids]
        )
        self.upper: FloatArray = np.asarray([m.jnt_range[i, 1] if m.jnt_limited[i] else np.inf for i in self.joint_ids])
        self.end_effectors = dict(config.end_effectors)
        for end_effector in self.end_effectors.values():
            self.body_id(end_effector.body)
        for group in config.joint_groups.values():
            if set(group) - set(self.joint_names):
                raise ValueError("joint_groups 包含模型中不存在的关节")
        self.home: FloatArray = m.qpos0.copy()
        if self.floating:
            if config.home.position is not None:
                self.home[:3] = config.home.position
            if config.home.quaternion is not None:
                self.home[3:7] = config.home.quaternion
        for field, values, joint_type in (
            ("joints_rad", config.home.joints_rad, mj.mjtJoint.mjJNT_HINGE),
            ("joints_m", config.home.joints_m, mj.mjtJoint.mjJNT_SLIDE),
        ):
            for name, value in values.items():
                if name not in self.joint_names:
                    raise ValueError(f"home.{field} 包含未知关节：{name}")
                index = self.joint_names.index(name)
                if m.jnt_type[self.joint_ids[index]] != joint_type:
                    raise ValueError(f"home.{field} 的单位与关节类型不匹配：{name}")
                self.home[self.qadr[index]] = value
        self.home = self.validate_qpos(self.home)
        self.environment = Environment(spec)

    @classmethod
    def load(cls, path: str | Path) -> "Robot":
        """读取本地机器人定义。实际模型、网格与本机配置不随源码分发。"""
        path = Path(path).expanduser().resolve()
        # JSON5 解析配置文本，Pydantic 随后校验字段与数值。
        raw = json5.loads(path.read_text(encoding="utf-8"), allow_duplicate_keys=False)
        return cls(RobotDefinition.model_validate(raw), config_path=path)

    def body_id(self, name: str) -> int:
        index = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, name)
        if index < 1:
            raise ValueError(f"找不到机器人连杆：{name}")
        return index

    def end_effector(self, name: str) -> EndEffector:
        if name not in self.end_effectors:
            raise ValueError(f"未知末端：{name}")
        return self.end_effectors[name]

    def validate_qpos(self, qpos: ArrayLike, *, limits: bool = True) -> FloatArray:
        qpos = np.asarray(qpos, dtype=float).copy()
        if qpos.shape != (self.model.nq,):
            raise ValueError(f"qpos 维度与模型不一致：需要 {self.model.nq} 个分量，实际形状为 {qpos.shape}")
        if not np.isfinite(qpos).all():
            raise ValueError(f"qpos 必须包含 {self.model.nq} 个有限数值")
        if self.floating:
            norm = np.linalg.norm(qpos[3:7])
            if not np.isfinite(norm) or norm < 1e-9:
                raise ValueError("根四元数不能为零")
            qpos[3:7] /= norm
        if limits and (np.any(qpos[self.qadr] < self.lower - 1e-8) or np.any(qpos[self.qadr] > self.upper + 1e-8)):
            raise ValueError("关节角度超出模型限位")
        return qpos

    def forward(self, qpos: ArrayLike, data: mj.MjData | None = None) -> mj.MjData:
        state = mj.MjData(self.model) if data is None else data
        state.qpos[:] = qpos
        mj.mj_kinematics(self.model, state)
        mj.mj_comPos(self.model, state)
        return state

    def pose(self, data: mj.MjData, name: str) -> tuple[FloatArray, FloatArray]:
        """返回同一末端坐标系的世界位置与朝向。"""
        end_effector = self.end_effector(name)
        body = self.body_id(end_effector.body)
        return data.xpos[body] + data.xmat[body].reshape(3, 3) @ end_effector.offset, data.xquat[body].copy()

    def center_of_mass(self, data: mj.MjData) -> FloatArray:
        return data.subtree_com[self.root_id].copy()

    def center_of_mass_jacobian(self, data: mj.MjData) -> FloatArray:
        jacobian = np.empty((3, self.model.nv))
        mj.mj_jacSubtreeCom(self.model, data, jacobian, self.root_id)
        return jacobian

    def selectable_joints(self, body: int) -> tuple[str, ...]:
        """网格归属最近的上游活动关节，刚性附属连杆不成为独立操作对象。"""
        while body:
            joints = tuple(
                self.model.joint(joint).name for joint in self.joint_ids if self.model.jnt_bodyid[joint] == body
            )
            if joints:
                return joints
            body = int(self.model.body_parentid[body])
        return ()

    def joint_pose(self, data: mj.MjData, name: str) -> tuple[FloatArray, FloatArray]:
        """关节控件位姿：Z 轴与真实关节轴一致，转环位于铰链锚点。"""
        m = self.model
        joint = self.joint_ids[self.joint_names.index(name)]
        z = data.xaxis[joint]
        reference = np.eye(3)[int(np.argmin(abs(z)))]
        x = np.cross(reference, z)
        x /= np.linalg.norm(x)
        frame = Rotation.from_matrix(np.column_stack((x, np.cross(z, x), z)))
        if m.jnt_type[joint] == mj.mjtJoint.mjJNT_HINGE:
            # 转环随当前关节转动，松开后不会跳回零角度；ref 对应模型 qpos0。
            address = m.jnt_qposadr[joint]
            frame *= Rotation.from_rotvec((0.0, 0.0, float(data.qpos[address] - m.qpos0[address])))
        orientation = quaternion(frame)
        position = (
            data.xpos[m.jnt_bodyid[joint]] if m.jnt_type[joint] == mj.mjtJoint.mjJNT_SLIDE else data.xanchor[joint]
        )
        return position.copy(), orientation

    def difference(self, a: ArrayLike, b: ArrayLike, dt: float = 1.0) -> FloatArray:
        out = np.empty(self.model.nv)
        mj.mj_differentiatePos(self.model, out, dt, np.asarray(a, dtype=float), np.asarray(b, dtype=float))
        return out

    def integrate(self, reference: ArrayLike, delta: ArrayLike) -> FloatArray:
        out = np.asarray(reference, dtype=float).copy()
        mj.mj_integratePos(self.model, out, np.asarray(delta, dtype=float), 1.0)
        return out
