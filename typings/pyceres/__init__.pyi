"""项目使用的 PyCeres 2.6 公共接口；上游 wheel 未提供类型声明。"""

from collections.abc import Sequence
from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray

type Array = NDArray[np.float64]

class CostFunction:
    def __init__(self) -> None: ...
    def set_num_residuals(self, value: int) -> None: ...
    def set_parameter_block_sizes(self, values: list[int]) -> None: ...
    def num_residuals(self) -> int: ...
    def Evaluate(
        self, parameters: Sequence[Array], residuals: Array, jacobians: Sequence[Array | None] | None
    ) -> bool: ...

class QuaternionManifold:
    def __init__(self) -> None: ...

class Problem:
    def __init__(self) -> None: ...
    def add_parameter_block(self, values: Array, size: int, manifold: QuaternionManifold | None = ...) -> None: ...
    def set_parameter_lower_bound(self, values: Array, index: int, value: float) -> None: ...
    def set_parameter_upper_bound(self, values: Array, index: int, value: float) -> None: ...
    def set_parameter_block_constant(self, values: Array) -> None: ...
    def is_parameter_block_constant(self, values: Array) -> bool: ...
    def add_residual_block(self, cost: CostFunction, loss: None, parameters: list[Array]) -> object: ...
    def num_residual_blocks(self) -> int: ...

class IterationSummary:
    iteration: int
    cost: float

class CallbackReturnType(Enum):
    SOLVER_ABORT = auto()
    SOLVER_CONTINUE = auto()

class IterationCallback:
    def __init__(self) -> None: ...
    def __call__(self, summary: IterationSummary) -> CallbackReturnType: ...

class LinearSolverType(Enum):
    SPARSE_NORMAL_CHOLESKY = auto()

class LoggingType(Enum):
    SILENT = auto()

class TerminationType(Enum):
    CONVERGENCE = auto()
    USER_SUCCESS = auto()

class SolverOptions:
    def __init__(self) -> None: ...
    max_num_iterations: int
    num_threads: int
    linear_solver_type: LinearSolverType
    minimizer_progress_to_stdout: bool
    logging_type: LoggingType
    callbacks: list[IterationCallback]
    function_tolerance: float
    gradient_tolerance: float
    parameter_tolerance: float

class SolverSummary:
    def __init__(self) -> None: ...
    final_cost: float
    termination_type: TerminationType
    message: str
    def IsSolutionUsable(self) -> bool: ...

def solve(options: SolverOptions, problem: Problem, summary: SolverSummary) -> None: ...
