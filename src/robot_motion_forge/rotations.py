"""项目 wxyz 四元数与 SciPy Rotation 之间的显式转换。"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation


def rotation_from_wxyz(wxyz: ArrayLike) -> Rotation:
    return Rotation.from_quat(np.asarray(wxyz, dtype=float), scalar_first=True)


def rotation_to_wxyz(rotation: Rotation) -> NDArray[np.float64]:
    return rotation.as_quat(scalar_first=True)
