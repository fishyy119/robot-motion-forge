"""项目 wxyz 四元数与 SO(3) 运算。"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation


def rotation_from_wxyz(wxyz: ArrayLike) -> Rotation:
    return Rotation.from_quat(np.asarray(wxyz, dtype=float), scalar_first=True)


def rotation_to_wxyz(rotation: Rotation) -> NDArray[np.float64]:
    return rotation.as_quat(scalar_first=True)


def skew(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    x, y, z = vector
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)


def log_jacobian_coefficient(angle2: NDArray[np.float64]) -> NDArray[np.float64]:
    # (1 - theta/2*cot(theta/2))/theta^2；零附近用级数避免相消。
    small = angle2 < 0.01
    safe = np.where(small, 1.0, angle2)
    half = np.sqrt(safe) / 2
    return np.where(small, 1 / 12 + angle2 / 720 + angle2**2 / 30240, (1 - half / np.tan(half)) / safe)


def inverse_right_jacobian(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    # J_r^{-1}(v) = I + [v]/2 + c(theta) * [v]^2，Log(SO(3)) 的角度在 [0, pi]。
    cross = skew(vector)
    coefficient = float(log_jacobian_coefficient(np.asarray(vector @ vector)))
    return np.eye(3) + 0.5 * cross + coefficient * (cross @ cross)


def log_end_tangent(delta: NDArray[np.float64], omega: NDArray[np.float64]) -> NDArray[np.float64]:
    """将空间角速度转换为左增量 Log 曲线的末端导数 J_l(delta)^(-1) omega。"""
    cross = np.cross(delta, omega)
    coefficient = log_jacobian_coefficient(np.sum(delta * delta, axis=-1))
    return omega - 0.5 * cross + coefficient[..., None] * np.cross(delta, cross)
