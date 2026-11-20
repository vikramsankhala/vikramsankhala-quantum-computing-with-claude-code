"""Module 15: teleportation, remote gates, circuit cutting and partitioning."""
from __future__ import annotations

import itertools
from math import ceil

import numpy as np

from .circuit import Circuit
from .gates import CNOT
from .operators import I2, X, Y, Z


def _extract(circ: Circuit, measured: dict[int, int]):
    """State of the unmeasured qubits after the measured ones collapsed to basis values."""
    psi = circ.state().reshape([2] * circ.n)
    idx = [slice(None)] * circ.n
    for q, v in measured.items():
        idx[q] = v
    out = psi[tuple(idx)].reshape(-1)
    return out / np.linalg.norm(out)


def teleport(psi: np.ndarray, m0: int, m1: int):
    """Teleport a qubit state; returns (Bob's state, probability of this outcome)."""
    c = Circuit(3, np.kron(psi, [1, 0, 0, 0]))
    c.h(1).cx(1, 2).cx(0, 1).h(0)
    p = (np.abs(c.state().reshape(2, 2, 2)[m0, m1]) ** 2).sum()
    c.measure(0, outcome=m0)
    c.measure(1, outcome=m1)
    if m1:
        c.x(2)
    if m0:
        c.z(2)
    return _extract(c, {0: m0, 1: m1}), float(p)


def remote_cnot(psi_ab: np.ndarray, m1: int, m2: int):
    """CNOT between qubit a (module A) and b (module B) using one shared Bell pair.

    Qubit order: a=0, b=1, e1=2 (module A), e2=3 (module B). Returns the state on
    (a, b) after the protocol and the probability of the outcome (m1, m2).
    """
    c = Circuit(4, np.kron(psi_ab, [1, 0, 0, 0]))
    c.h(2).cx(2, 3)          # the shared Bell pair
    c.cx(0, 2)               # module A: entangle control with its half
    p1 = float(np.sum(np.abs(c.state().reshape([2] * 4)[:, :, m1, :]) ** 2))
    c.measure(2, outcome=m1)
    if m1:
        c.x(3)               # module B: classical correction
    c.cx(3, 1)               # module B: act on the target
    c.h(3)
    p2 = float(np.sum(np.abs(c.state().reshape([2] * 4)[:, :, m1, m2]) ** 2))  # already conditional: measure() renormalises
    c.measure(3, outcome=m2)
    if m2:
        c.z(0)               # module A: classical correction
    return _extract(c, {2: m1, 3: m2}), p1 * p2


def local_cnot(psi_ab: np.ndarray) -> np.ndarray:
    return CNOT @ psi_ab


# ----------------------------------------------------------- wire cutting
def cut_reconstruction(rho: np.ndarray) -> np.ndarray:
    """rho = 1/2 sum_P Tr(P rho) P: the identity channel as a sum of measure-and-prepare terms."""
    return 0.5 * sum(np.trace(P @ rho) * P for P in (I2, X, Y, Z))


def shot_overhead(wire_cuts: int = 0, cnot_cuts: int = 0) -> int:
    """Commonly quoted sampling overhead: 16 per wire cut (kappa=4, squared) and 9
    per CNOT gate cut (kappa=3, squared). Literature values, not derived here."""
    return 16 ** wire_cuts * 9 ** cnot_cuts


def min_cut_bipartition(n: int, edges):
    """Balanced bipartition of n qubits minimising the number of cut two-qubit gates."""
    best = None
    for A in itertools.combinations(range(n), n // 2):
        S = set(A)
        cut = sum((i in S) != (j in S) for i, j in edges)
        if best is None or cut < best[0]:
            best = (cut, sorted(S))
    return best


def modules_needed(n_qubits: int, module_size: int) -> int:
    return ceil(n_qubits / module_size)
