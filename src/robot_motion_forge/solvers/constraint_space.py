"""将线性硬约束化为可行增量及其零空间，供 IK 与奇异姿态探测共用。"""

from dataclasses import dataclass

import numpy as np

from robot_motion_forge.robot.model import FloatArray

LINEAR_CONSTRAINT_TOLERANCE = 1e-9


class InconsistentConstraints(ValueError):
    """当前线性化中的硬约束没有共同解。"""


@dataclass(frozen=True)
class ConstraintSpace:
    # J Δq = b 的可行增量为 Δq = offset + basis z；basis 的列正交归一。
    offset: FloatArray
    basis: FloatArray

    @classmethod
    def from_equalities(cls, jacobian: FloatArray, feedback: FloatArray) -> "ConstraintSpace":
        size = jacobian.shape[1]
        if not len(jacobian):
            return cls(np.zeros(size), np.eye(size))
        left, singular, right = np.linalg.svd(jacobian, full_matrices=True)
        # 只消去机器精度下的相关行，保留模型中虽弱但真实存在的约束方向。
        threshold = np.finfo(float).eps * max(jacobian.shape) * singular[0] if len(singular) else 0.0
        rank = int(np.count_nonzero(singular > threshold))
        components = left[:, :rank].T @ feedback
        residual = feedback - left[:, :rank] @ components
        if np.linalg.norm(residual, ord=np.inf) > LINEAR_CONSTRAINT_TOLERANCE * max(
            1.0, float(np.linalg.norm(feedback, ord=np.inf))
        ):
            raise InconsistentConstraints("当前线性化中的固定约束不一致")
        offset = right[:rank].T @ (components / singular[:rank])
        return cls(offset, right[rank:].T)
