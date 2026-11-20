"""项目 wxyz 四元数与 SO(3) 运算。"""

import math

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation

from .array_types import FloatArray


def normalize_wxyz(wxyz: ArrayLike, *, preserve_unit: bool = False) -> FloatArray:
    """归一化单个 wxyz 四元数；可保留容差内的单位四元数原值。"""
    quaternion = np.asarray(wxyz, dtype=float)
    if quaternion.shape != (4,):
        raise ValueError("四元数必须包含四个 wxyz 分量")
    norm = math.hypot(*quaternion)
    if not math.isfinite(norm) or norm < 1e-9:
        raise ValueError("wxyz 四元数必须为非零有限数值")
    if preserve_unit and abs(norm - 1.0) <= 1e-12:
        return quaternion
    return quaternion / norm


def rotation_from_wxyz(wxyz: ArrayLike) -> Rotation:
    return Rotation.from_quat(np.asarray(wxyz, dtype=float), scalar_first=True)


def rotation_to_wxyz(rotation: Rotation) -> FloatArray:
    return rotation.as_quat(scalar_first=True)


def skew(vector: FloatArray) -> FloatArray:
    x, y, z = vector
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)


def log_jacobian_coefficient(angle2: FloatArray) -> FloatArray:
    # (1 - theta/2*cot(theta/2))/theta^2；零附近用级数避免相消。
    small = angle2 < 0.01
    safe = np.where(small, 1.0, angle2)
    half = np.sqrt(safe) / 2
    return np.where(small, 1 / 12 + angle2 / 720 + angle2**2 / 30240, (1 - half / np.tan(half)) / safe)


def inverse_right_jacobian(vector: FloatArray) -> FloatArray:
    # J_r^{-1}(v) = I + [v]/2 + c(theta) * [v]^2，Log(SO(3)) 的角度在 [0, pi]。
    cross = skew(vector)
    coefficient = float(log_jacobian_coefficient(np.asarray(vector @ vector)))
    return np.eye(3) + 0.5 * cross + coefficient * (cross @ cross)


def log_end_tangent(delta: FloatArray, omega: FloatArray) -> FloatArray:
    """将空间角速度转换为左增量 Log 曲线的末端导数 J_l(delta)^(-1) omega。"""
    cross = np.cross(delta, omega)
    coefficient = log_jacobian_coefficient(np.sum(delta * delta, axis=-1))
    return omega - 0.5 * cross + coefficient[..., None] * np.cross(delta, cross)


def exp_spatial_velocity(vector: FloatArray, tangent: FloatArray) -> FloatArray:
    """将旋转向量的导数转换为空间角速度 J_l(vector) tangent。"""
    angle2 = np.sum(vector * vector, axis=-1)
    angle = np.sqrt(angle2)
    small = angle2 < 0.01
    # J_l(v) = I + (1-cos(theta))/theta² [v] + (theta-sin(theta))/theta³ [v]²。
    first = 0.5 * np.sinc(angle / (2 * np.pi)) ** 2
    second = np.where(
        small,
        1 / 6 - angle2 / 120 + angle2**2 / 5040 - angle2**3 / 362880,
        (1 - np.sinc(angle / np.pi)) / np.where(small, 1.0, angle2),
    )
    cross = np.cross(vector, tangent)
    return tangent + first[..., None] * cross + second[..., None] * np.cross(vector, cross)


def log_end_tangent_jacobian(delta: FloatArray, omega: FloatArray) -> FloatArray:
    """末端 Log 导数 J_l(delta)^(-1) omega 对 delta 的雅可比。"""
    angle2 = float(delta @ delta)
    c = float(log_jacobian_coefficient(np.asarray(angle2)))
    if angle2 < 0.01:
        dc = 1 / 720 + 2 * angle2 / 30240 + angle2**2 / 403200
    else:
        half = np.sqrt(angle2) / 2
        dc = (-1 / np.tan(half) + half / np.sin(half) ** 2) / (8 * half * angle2) - c / angle2
    dot = float(delta @ omega)
    cross2 = delta * dot - angle2 * omega
    return (
        0.5 * skew(omega)
        + c * (dot * np.eye(3) + np.outer(delta, omega) - 2 * np.outer(omega, delta))
        + 2 * dc * np.outer(cross2, delta)
    )
