"""Module 4: quantum gates."""
from __future__ import annotations

import numpy as np

from .operators import I2, X, Y, Z

H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
SDG = S.conj().T
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)

CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
CZ = np.diag([1, 1, 1, -1]).astype(complex)
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)


def rx(theta: float) -> np.ndarray:
    return np.cos(theta / 2) * I2 - 1j * np.sin(theta / 2) * X


def ry(theta: float) -> np.ndarray:
    return np.cos(theta / 2) * I2 - 1j * np.sin(theta / 2) * Y


def rz(theta: float) -> np.ndarray:
    return np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex)


def phase(phi: float) -> np.ndarray:
    return np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=complex)


def cphase(phi: float) -> np.ndarray:
    return np.diag([1, 1, 1, np.exp(1j * phi)]).astype(complex)


def controlled(U: np.ndarray) -> np.ndarray:
    """Controlled-U on (control, target) with the control as the first qubit."""
    d = U.shape[0]
    out = np.eye(2 * d, dtype=complex)
    out[d:, d:] = U
    return out


TOFFOLI = controlled(CNOT)


def zyz_decompose(U: np.ndarray):
    """Return (alpha, beta, gamma, delta) with U = e^{i alpha} Rz(beta) Ry(gamma) Rz(delta).

    Every single-qubit gate is three rotations and a global phase, which is
    why a small gate set is enough in hardware.
    """
    alpha = np.angle(np.linalg.det(U)) / 2
    V = U * np.exp(-1j * alpha)
    gamma = 2 * np.arctan2(abs(V[1, 0]), abs(V[0, 0]))
    s = -2 * np.angle(V[0, 0]) if abs(V[0, 0]) > 1e-12 else 0.0
    d = 2 * np.angle(V[1, 0]) if abs(V[1, 0]) > 1e-12 else 0.0
    return float(alpha), float((s + d) / 2), float(gamma), float((s - d) / 2)


def from_zyz(alpha: float, beta: float, gamma: float, delta: float) -> np.ndarray:
    return np.exp(1j * alpha) * rz(beta) @ ry(gamma) @ rz(delta)


def random_unitary(d: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Haar-random unitary via QR of a complex Gaussian matrix."""
    rng = rng or np.random.default_rng()
    z = (rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / abs(np.diag(r)))


def clifford_check(U: np.ndarray) -> bool:
    """True if U maps every single-qubit Pauli X, Z (tensor identity) to a Pauli string."""
    n = int(np.log2(U.shape[0]))
    from .operators import pauli_string
    import itertools

    letters = "IXYZ"
    paulis = {"".join(p): pauli_string("".join(p)) for p in itertools.product(letters, repeat=n)}
    for g in ("X" + "I" * (n - 1), "Z" + "I" * (n - 1)):
        out = U @ paulis[g] @ U.conj().T
        if not any(np.allclose(out, s * m) for m in paulis.values() for s in (1, -1)):
            return False
    return True
