"""Module 14: the hybrid quantum-classical loop and its cost model."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


def shots_for_precision(variance: float, epsilon: float) -> int:
    """Shots needed to estimate an expectation value to standard error epsilon."""
    return int(np.ceil(variance / epsilon ** 2))


def amdahl_speedup(quantum_fraction: float, quantum_speedup: float) -> float:
    """End-to-end speedup when only a fraction of the workload is accelerated."""
    return 1 / ((1 - quantum_fraction) + quantum_fraction / quantum_speedup)


def iteration_time(shots: int, shot_time_s: float, classical_s: float, queue_s: float = 0.0) -> float:
    return shots * shot_time_s + classical_s + queue_s


@dataclass
class HybridLoop:
    """A classical optimiser driving a quantum evaluator (a QPU or a simulator)."""

    evaluate: Callable[[np.ndarray], float]
    history: list = field(default_factory=list)
    calls: int = 0

    def objective(self, params) -> float:
        self.calls += 1
        v = float(self.evaluate(np.asarray(params)))
        self.history.append(v)
        return v

    def run(self, x0, method: str = "COBYLA", maxiter: int = 200):
        from scipy.optimize import minimize

        res = minimize(self.objective, x0, method=method, options={"maxiter": maxiter})
        return res.x, float(res.fun)
