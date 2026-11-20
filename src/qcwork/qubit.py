"""Module 3: qubits, the Bloch sphere, density matrices and entanglement."""
from __future__ import annotations

import numpy as np

from .operators import X, Y, Z, expectation

ZERO = np.array([1, 0], dtype=complex)
ONE = np.array([0, 1], dtype=complex)
PLUS = (ZERO + ONE) / np.sqrt(2)
MINUS = (ZERO - ONE) / np.sqrt(2)


def state_from_bloch(theta: float, phi: float) -> np.ndarray:
    """|psi> = cos(theta/2)|0> + e^{i phi} sin(theta/2)|1>."""
    return np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)], dtype=complex)


def bloch_vector(psi: np.ndarray) -> np.ndarray:
    """(<X>, <Y>, <Z>) of a single-qubit pure state."""
    return np.array([expectation(psi, X), expectation(psi, Y), expectation(psi, Z)])


def density_matrix(psi: np.ndarray) -> np.ndarray:
    return np.outer(psi, psi.conj())


def bloch_vector_from_rho(rho: np.ndarray) -> np.ndarray:
    return np.array([np.trace(rho @ P).real for P in (X, Y, Z)])


def purity(rho: np.ndarray) -> float:
    """Tr(rho^2): 1 for a pure state, 1/d for the maximally mixed state."""
    return float(np.trace(rho @ rho).real)


def bell_state(k: int = 0) -> np.ndarray:
    """The four Bell states: 0=Phi+, 1=Phi-, 2=Psi+, 3=Psi-."""
    s = 1 / np.sqrt(2)
    return [
        np.array([s, 0, 0, s], dtype=complex),
        np.array([s, 0, 0, -s], dtype=complex),
        np.array([0, s, s, 0], dtype=complex),
        np.array([0, s, -s, 0], dtype=complex),
    ][k]


def ghz_state(n: int) -> np.ndarray:
    psi = np.zeros(2 ** n, dtype=complex)
    psi[0] = psi[-1] = 1 / np.sqrt(2)
    return psi


def reduced_density_matrix(psi: np.ndarray, keep: list[int], n: int) -> np.ndarray:
    """Partial trace over all qubits not in `keep` (qubit 0 = leftmost)."""
    t = psi.reshape([2] * n)
    rest = [q for q in range(n) if q not in keep]
    t = np.transpose(t, list(keep) + rest).reshape(2 ** len(keep), -1)
    return t @ t.conj().T


def schmidt_coefficients(psi: np.ndarray, n_a: int, n: int) -> np.ndarray:
    """Schmidt coefficients across the cut first n_a qubits | the rest."""
    return np.linalg.svd(psi.reshape(2 ** n_a, 2 ** (n - n_a)), compute_uv=False)


def entanglement_entropy(psi: np.ndarray, n_a: int, n: int) -> float:
    """Von Neumann entropy (bits) of the first n_a qubits."""
    s = schmidt_coefficients(psi, n_a, n) ** 2
    s = s[s > 1e-12]
    return float(-(s * np.log2(s)).sum())


def is_product_state(psi: np.ndarray, n_a: int, n: int, tol: float = 1e-9) -> bool:
    """A pure state is a product across a cut iff its Schmidt rank is 1."""
    return int((schmidt_coefficients(psi, n_a, n) > tol).sum()) == 1
