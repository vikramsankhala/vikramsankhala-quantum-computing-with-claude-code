"""Module 1: Hilbert space, state vectors, inner products and bases.

Conventions used throughout the repository
------------------------------------------
* A state is a one-dimensional complex NumPy array (a ket |psi>).
* inner(a, b) is <a|b>: conjugate-linear in the first argument.
* Qubit 0 is the most significant (leftmost) bit of a basis label, as in the
  textbook |q0 q1 ... q(n-1)> notation. Qiskit uses the opposite order; the
  cross-framework tests handle the conversion.
"""
from __future__ import annotations

from functools import reduce

import numpy as np


def ket(*amplitudes, normalize_state: bool = False) -> np.ndarray:
    """Build a ket from amplitudes, optionally normalised."""
    v = np.array(amplitudes, dtype=complex)
    return normalize(v) if normalize_state else v


def basis_vector(index: int, dim: int) -> np.ndarray:
    """Standard basis vector e_index in C^dim."""
    e = np.zeros(dim, dtype=complex)
    e[index] = 1.0
    return e


def inner(a: np.ndarray, b: np.ndarray) -> complex:
    """<a|b>."""
    return complex(np.vdot(a, b))


def norm(a: np.ndarray) -> float:
    """||a|| = sqrt(<a|a>)."""
    return float(np.sqrt(inner(a, a).real))


def normalize(a: np.ndarray) -> np.ndarray:
    n = norm(a)
    if n == 0:
        raise ValueError("cannot normalise the zero vector")
    return a / n


def is_normalized(a: np.ndarray, tol: float = 1e-12) -> bool:
    return abs(norm(a) - 1.0) < tol


def outer(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """|a><b|."""
    return np.outer(a, np.conj(b))


def projector(a: np.ndarray) -> np.ndarray:
    """Projector |a><a| onto the (normalised) vector a."""
    a = normalize(a)
    return outer(a, a)


def gram_schmidt(vectors, tol: float = 1e-12) -> list[np.ndarray]:
    """Orthonormalise a list of vectors, dropping linearly dependent ones."""
    result: list[np.ndarray] = []
    for v in vectors:
        w = np.array(v, dtype=complex)
        for u in result:
            w = w - inner(u, w) * u
        if norm(w) > tol:
            result.append(w / norm(w))
    return result


def is_orthonormal(vectors, tol: float = 1e-10) -> bool:
    """True if the Gram matrix of the vectors is the identity."""
    V = np.column_stack(vectors)
    return bool(np.allclose(V.conj().T @ V, np.eye(V.shape[1]), atol=tol))


def coefficients(psi: np.ndarray, basis) -> np.ndarray:
    """Expansion coefficients c_k = <b_k|psi> in an orthonormal basis."""
    return np.array([inner(b, psi) for b in basis])


def reconstruct(coeffs, basis) -> np.ndarray:
    """psi = sum_k c_k |b_k>."""
    return sum(c * b for c, b in zip(coeffs, basis))


def tensor(*vectors: np.ndarray) -> np.ndarray:
    """Tensor (Kronecker) product of kets."""
    return reduce(np.kron, vectors)


def fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """|<a|b>|^2 for normalised pure states."""
    return float(abs(inner(a, b)) ** 2)


def cauchy_schwarz_gap(a: np.ndarray, b: np.ndarray) -> float:
    """||a|| ||b|| - |<a|b>|; non-negative for any two vectors."""
    return norm(a) * norm(b) - abs(inner(a, b))


def computational_basis(n_qubits: int) -> list[np.ndarray]:
    """The 2^n computational basis kets |00..0>, |00..1>, ..."""
    d = 2 ** n_qubits
    return [basis_vector(i, d) for i in range(d)]


def dimension_of_n_qubits(n_qubits: int) -> int:
    """Hilbert-space dimension of n qubits: 2^n."""
    return 2 ** n_qubits
