"""由 MuJoCo 等式约束重建从动关节及其切空间映射。"""

from threading import local

import mujoco as mj
import numpy as np
from numpy.typing import NDArray

from robot_motion_forge.utils.array_types import FloatArray
from robot_motion_forge.utils.rotations import inverse_right_jacobian

_RESIDUAL_TOLERANCE = 1e-10
_RANK_TOLERANCE = 1e-9
_LIMIT_TOLERANCE = 1e-8


class ClosureError(ValueError):
    """当前主动坐标在选定的装配分支上无法重建。"""


def _infer_passive_joints(model: mj.MjModel, jacobian: FloatArray) -> list[int]:
    transmission = model.actuator_trntype
    direct = (transmission == mj.mjtTrn.mjTRN_JOINT) | (transmission == mj.mjtTrn.mjTRN_JOINTINPARENT)
    if not np.all(direct):
        raise ClosureError("闭链自动推断要求直接关节驱动，模型含间接执行器传动")
    driven = set(model.actuator_trnid[:, 0].tolist())
    singular = np.linalg.svd(jacobian, compute_uv=False)
    threshold = float(singular[0]) * _RANK_TOLERANCE if len(singular) else 0.0
    rank = int(np.count_nonzero(singular > threshold))
    passive: list[int] = []
    joint_rank = 0
    for joint in range(model.njnt):
        if joint in driven or model.jnt_type[joint] == mj.mjtJoint.mjJNT_FREE:
            continue
        block = jacobian[:, model.dof_jntid == joint]
        constraint_rank = int(np.linalg.matrix_rank(block, tol=threshold))
        if constraint_rank:
            passive.append(joint)
            joint_rank += constraint_rank
    if not passive:
        raise ClosureError("默认姿态下没有能参与闭合的从动关节")
    columns = np.isin(model.dof_jntid, passive)
    if np.linalg.matrix_rank(jacobian[:, columns], tol=threshold) != rank:
        raise ClosureError("默认姿态下候选从动关节无法覆盖全部闭合约束方向")
    # 每个关节内部的自由方向可由默认姿态确定。跨关节的额外自由方向使自动划分有歧义。
    if joint_rank != rank:
        raise ClosureError("候选从动关节之间存在未约束的自由方向，无法唯一划分")
    return passive


class _Workspace(local):
    def __init__(self, model: mj.MjModel) -> None:
        self.data = mj.MjData(model)


class PassiveClosure:
    def __init__(self, model: mj.MjModel, home: FloatArray) -> None:
        self.model = model
        self._workspace = _Workspace(model)
        self._reference = home.copy()
        _, jacobian = self._equalities(home)
        self.joint_ids = tuple(_infer_passive_joints(model, jacobian))
        self._qadr: list[int] = []
        dofs: list[int] = []
        self._ball_rows: list[slice] = []
        for joint in self.joint_ids:
            qadr = int(model.jnt_qposadr[joint])
            dof = int(model.jnt_dofadr[joint])
            if model.jnt_type[joint] == mj.mjtJoint.mjJNT_BALL:
                self._ball_rows.append(slice(len(dofs), len(dofs) + 3))
                self._qadr.extend(range(qadr, qadr + 4))
                dofs.extend(range(dof, dof + 3))
            else:
                self._qadr.append(qadr)
                dofs.append(dof)
        self.independent: NDArray[np.intp] = np.setdiff1d(np.arange(model.nv), dofs)
        self.passive = np.asarray(dofs, dtype=int)
        self._independent_qadr: NDArray[np.intp] = np.setdiff1d(np.arange(model.nq), self._qadr)
        _, singular, right = np.linalg.svd(jacobian[:, self.passive], full_matrices=False)
        rank = int(np.count_nonzero(singular > singular[0] * _RANK_TOLERANCE)) if len(singular) else 0
        # 在默认姿态固定闭合约束未决定的从动方向，例如球铰连杆的轴向自转。
        self._basis = right[:rank].T.copy()

    def _equalities(self, pose: FloatArray) -> tuple[FloatArray, FloatArray]:
        model, data = self.model, self._workspace.data
        data.qpos[:] = pose
        mj.mj_kinematics(model, data)
        mj.mj_comPos(model, data)
        mj.mj_tendon(model, data)
        mj.mj_makeConstraint(model, data)
        rows = np.flatnonzero(data.efc_type == mj.mjtConstraint.mjCNSTR_EQUALITY)
        residual = data.efc_pos[rows]
        if not mj.mj_isSparse(model):
            return residual, data.efc_J.reshape(data.nefc, model.nv)[rows]
        jacobian = np.empty((len(rows), model.nv))
        selector = np.zeros(data.nefc)
        for index, row in enumerate(rows):
            selector[row] = 1
            mj.mj_mulJacTVec(model, data, jacobian[index], selector)
            selector[row] = 0
        return residual, jacobian

    def _passive_tangent(self, offset: FloatArray) -> FloatArray:
        tangent = self._basis.copy()
        for rows in self._ball_rows:
            rotation = offset[rows]
            if np.linalg.norm(rotation) >= np.pi:
                raise ClosureError("从动球关节超出默认姿态的局部旋转范围")
            tangent[rows] = np.linalg.solve(inverse_right_jacobian(rotation), tangent[rows])
        return tangent

    def _check_limits(self, pose: FloatArray) -> None:
        for joint in self.joint_ids:
            if not self.model.jnt_limited[joint]:
                continue
            qadr = int(self.model.jnt_qposadr[joint])
            lower, upper = self.model.jnt_range[joint]
            value = pose[qadr]
            if self.model.jnt_type[joint] == mj.mjtJoint.mjJNT_BALL:
                value = 2 * np.arctan2(np.linalg.norm(pose[qadr + 1 : qadr + 4]), abs(value))
            if value < lower - _LIMIT_TOLERANCE or value > upper + _LIMIT_TOLERANCE:
                raise ClosureError(f"闭合姿态的从动关节超出限位：{self.model.joint(joint).name}")

    def reconstruct(
        self, pose: FloatArray, *, jacobian: bool = False, limits: bool = True
    ) -> tuple[FloatArray, FloatArray | None]:
        base = pose.copy()
        base[self._qadr] = self._reference[self._qadr]
        result = base.copy()
        offset = np.zeros(len(self.passive))
        delta = np.zeros(self.model.nv)
        residual, closure = self._equalities(result)
        for _ in range(24):
            tangent = self._passive_tangent(offset)
            passive = closure[:, self.passive] @ tangent
            error = float(np.linalg.norm(residual))
            if error <= _RESIDUAL_TOLERANCE:
                active = closure[:, self.independent]
                response, _, rank, _ = np.linalg.lstsq(passive, -active, rcond=_RANK_TOLERANCE)
                if rank < self._basis.shape[1] or np.linalg.norm(passive @ response + active) > _RESIDUAL_TOLERANCE:
                    raise ClosureError("闭合约束奇异，或当前主动关节无法独立运动")
                if limits:
                    self._check_limits(result)
                if not jacobian:
                    return result, None
                lift = np.eye(self.model.nv)
                lift[self.passive] = 0
                lift[np.ix_(self.passive, self.independent)] = tangent @ response
                return result, lift

            step, _, rank, _ = np.linalg.lstsq(passive, -residual, rcond=_RANK_TOLERANCE)
            if rank < self._basis.shape[1]:
                raise ClosureError("从动关节的闭合雅可比降秩")
            direction = self._basis @ step
            for scale in (1, 0.5, 0.25, 0.125, 0.0625, 0.03125):
                trial = base.copy()
                delta[self.passive] = offset + scale * direction
                mj.mj_integratePos(self.model, trial, delta, 1.0)
                trial[self._independent_qadr] = base[self._independent_qadr]
                trial_residual, trial_jacobian = self._equalities(trial)
                if np.linalg.norm(trial_residual) < error:
                    result, residual, closure = trial, trial_residual, trial_jacobian
                    offset = delta[self.passive]
                    break
            else:
                break
        raise ClosureError(f"从动关节无法闭合，约束残差 {np.linalg.norm(residual):.3g}")
