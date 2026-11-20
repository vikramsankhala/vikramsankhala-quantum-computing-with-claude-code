"""Module 2: operators, measurement and the Schrodinger equation.

Natural units hbar = 1, so the Schrodinger equation reads
    i d|psi>/dt = H |psi>,   |psi(t)> = exp(-i H t) |psi(0)>.
"""
from __future__ import annotations

from functools import reduce

import numpy as np
from scipy.linalg import expm

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULIS = {"I": I2, "X": X, "Y": Y, "Z": Z}


def pauli_string(s: str) -> np.ndarray:
    """Matrix of a Pauli string such as 'XZ' (leftmost letter = qubit 0)."""
    return reduce(np.kron, [PAULIS[c] for c in s])


def dagger(A: np.ndarray) -> np.ndarray:
    return A.conj().T


def is_hermitian(A: np.ndarray, tol: float = 1e-10) -> bool:
    return bool(np.allclose(A, dagger(A), atol=tol))


def is_unitary(U: np.ndarray, tol: float = 1e-10) -> bool:
    return bool(np.allclose(dagger(U) @ U, np.eye(U.shape[0]), atol=tol))


def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B - B @ A


def anticommutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B + B @ A


def expectation(psi: np.ndarray, A: np.ndarray) -> float:
    """<psi|A|psi>, real for Hermitian A."""
    return float(np.vdot(psi, A @ psi).real)


def variance(psi: np.ndarray, A: np.ndarray) -> float:
    return expectation(psi, A @ A) - expectation(psi, A) ** 2


def spectral_decomposition(A: np.ndarray, tol: float = 1e-9):
    """Eigenvalues with their projectors for a Hermitian operator.

    Returns a list of (eigenvalue, projector) with degenerate eigenvalues
    grouped, so that A = sum_k lambda_k P_k and sum_k P_k = I.
    """
    if not is_hermitian(A):
        raise ValueError("operator is not Hermitian")
    vals, vecs = np.linalg.eigh(A)
    groups: list[tuple[float, np.ndarray]] = []
    for k, lam in enumerate(vals):
        v = vecs[:, k]
        P = np.outer(v, v.conj())
        if groups and abs(groups[-1][0] - lam) < tol:
            groups[-1] = (groups[-1][0], groups[-1][1] + P)
        else:
            groups.append((float(lam), P))
    return groups


def born_probabilities(psi: np.ndarray, A: np.ndarray):
    """Born rule: list of (eigenvalue, probability) for measuring A on psi."""
    return [(lam, float(np.vdot(psi, P @ psi).real)) for lam, P in spectral_decomposition(A)]


def measure(psi: np.ndarray, A: np.ndarray, rng: np.random.Generator | None = None):
    """Projective measurement: returns (outcome, post-measurement state)."""
    rng = rng or np.random.default_rng()
    groups = spectral_decomposition(A)
    probs = np.array([float(np.vdot(psi, P @ psi).real) for _, P in groups])
    probs = probs / probs.sum()
    k = int(rng.choice(len(groups), p=probs))
    lam, P = groups[k]
    post = P @ psi
    return lam, post / np.linalg.norm(post)


def evolve(psi0: np.ndarray, H: np.ndarray, t: float) -> np.ndarray:
    """Exact solution of the Schrodinger equation for a time-independent H."""
    if not is_hermitian(H):
        raise ValueError("Hamiltonian must be Hermitian")
    return expm(-1j * H * t) @ psi0


def schrodinger_rk4(psi0: np.ndarray, H: np.ndarray, t: float, steps: int = 1000) -> np.ndarray:
    """Integrate i d psi/dt = H psi with a fourth-order Runge-Kutta scheme."""
    f = lambda v: -1j * (H @ v)
    dt = t / steps
    psi = psi0.astype(complex).copy()
    for _ in range(steps):
        k1 = f(psi)
        k2 = f(psi + 0.5 * dt * k1)
        k3 = f(psi + 0.5 * dt * k2)
        k4 = f(psi + dt * k3)
        psi = psi + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return psi


def time_evolution_unitary(H: np.ndarray, t: float) -> np.ndarray:
    """U(t) = exp(-iHt): why every closed-system evolution is a unitary gate."""
    return expm(-1j * H * t)


def uncertainty_product(psi: np.ndarray, A: np.ndarray, B: np.ndarray):
    """Return (sigma_A * sigma_B, |<[A,B]>| / 2): the first is never smaller."""
    lhs = float(np.sqrt(variance(psi, A) * variance(psi, B)))
    rhs = 0.5 * abs(np.vdot(psi, commutator(A, B) @ psi))
    return lhs, float(rhs)


def ground_state(H: np.ndarray):
    """Lowest eigenvalue and its eigenvector."""
    vals, vecs = np.linalg.eigh(H)
    return float(vals[0]), vecs[:, 0]
