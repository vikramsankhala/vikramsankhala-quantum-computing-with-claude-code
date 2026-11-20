"""Module 10: stabilizer codes, the repetition code, surface-code memory and
resource estimation."""
from __future__ import annotations

from math import comb

import numpy as np


# ------------------------------------------------------------ repetition code
def repetition_logical_error_exact(p: float, d: int) -> float:
    """P(majority of d bits flips) for odd d."""
    return sum(comb(d, k) * p ** k * (1 - p) ** (d - k) for k in range(d // 2 + 1, d + 1))


def repetition_logical_error_mc(p: float, d: int, shots: int = 200_000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    flips = rng.random((shots, d)) < p
    return float((flips.sum(axis=1) > d // 2).mean())


# ---------------------------------------------------------- stabilizer algebra
def pauli_to_symplectic(s: str) -> np.ndarray:
    """Pauli string -> binary vector (x | z) of length 2n."""
    x = [1 if c in "XY" else 0 for c in s]
    z = [1 if c in "ZY" else 0 for c in s]
    return np.array(x + z, dtype=np.uint8)


def symplectic_commute(a: str, b: str) -> bool:
    """Two Pauli strings commute iff their symplectic product is 0 (mod 2)."""
    n = len(a)
    va, vb = pauli_to_symplectic(a), pauli_to_symplectic(b)
    return int((va[:n] @ vb[n:] + va[n:] @ vb[:n]) % 2) == 0


def gf2_rank(M: np.ndarray) -> int:
    M = M.copy() % 2
    rank = 0
    rows, cols = M.shape
    for c in range(cols):
        pivot = next((r for r in range(rank, rows) if M[r, c]), None)
        if pivot is None:
            continue
        M[[rank, pivot]] = M[[pivot, rank]]
        for r in range(rows):
            if r != rank and M[r, c]:
                M[r] ^= M[rank]
        rank += 1
    return rank


def steane_stabilizers() -> list[str]:
    """Generators of the [[7,1,3]] Steane code from the Hamming parity checks."""
    H = ["0001111", "0110011", "1010101"]
    xs = ["".join("X" if b == "1" else "I" for b in row) for row in H]
    zs = ["".join("Z" if b == "1" else "I" for b in row) for row in H]
    return xs + zs


def code_parameters(stabilizers: list[str]):
    """Return (n, k) = (physical qubits, logical qubits) of a stabilizer code."""
    n = len(stabilizers[0])
    if not all(symplectic_commute(a, b) for a in stabilizers for b in stabilizers):
        raise ValueError("stabilizers must commute")
    r = gf2_rank(np.array([pauli_to_symplectic(s) for s in stabilizers]))
    return n, n - r


# --------------------------------------------------------- surface-code memory
def surface_code_logical_error(d: int, p: float, shots: int = 20_000, seed: int = 1) -> float:
    """Logical error rate per shot of a rotated surface-code memory (d rounds).

    Uses Stim to sample and PyMatching to decode; circuit-level depolarising
    noise with strength p.
    """
    import pymatching
    import stim

    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=d,
        rounds=d,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
        before_round_data_depolarization=p,
    )
    dem = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    sampler = circuit.compile_detector_sampler(seed=seed)
    det, obs = sampler.sample(shots, separate_observables=True)
    pred = matcher.decode_batch(det)
    return float((pred != obs).any(axis=1).mean())


# ------------------------------------------------------------ resource model
def logical_error_per_round(d: int, p_phys: float, p_th: float = 0.01, A: float = 0.1) -> float:
    """Rule-of-thumb model: A (p / p_th)^((d+1)/2). Not a vendor estimator."""
    return A * (p_phys / p_th) ** ((d + 1) / 2)


def distance_for_target(p_phys: float, target_total: float, logical_qubits: int, rounds: float,
                        p_th: float = 0.01, A: float = 0.1) -> int:
    """Smallest odd distance whose total failure probability meets the target."""
    d = 3
    while logical_qubits * rounds * logical_error_per_round(d, p_phys, p_th, A) > target_total:
        d += 2
        if d > 101:
            raise ValueError("physical error rate too close to threshold")
    return d


def physical_qubits(logical_qubits: int, d: int, overhead: float = 1.0) -> int:
    """Rotated surface code: 2 d^2 - 1 physical qubits per logical qubit, times an
    overhead factor for routing and magic-state factories."""
    return int(np.ceil(logical_qubits * (2 * d * d - 1) * overhead))
