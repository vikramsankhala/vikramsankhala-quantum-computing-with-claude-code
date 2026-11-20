"""Module 9: noise channels, error mitigation and benchmarking."""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from .operators import I2, X, Y, Z


def apply_kraus(rho: np.ndarray, kraus, qubit: int, n: int) -> np.ndarray:
    """Apply a single-qubit channel {K_i} to one qubit of an n-qubit density matrix."""
    out = np.zeros_like(rho)
    for K in kraus:
        full = np.eye(1, dtype=complex)
        for q in range(n):
            full = np.kron(full, K if q == qubit else I2)
        out += full @ rho @ full.conj().T
    return out


def depolarizing(p: float):
    return [np.sqrt(1 - 3 * p / 4) * I2, np.sqrt(p / 4) * X, np.sqrt(p / 4) * Y, np.sqrt(p / 4) * Z]


def bit_flip(p: float):
    return [np.sqrt(1 - p) * I2, np.sqrt(p) * X]


def amplitude_damping(gamma: float):
    return [np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex),
            np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)]


def phase_damping(lam: float):
    return [np.array([[1, 0], [0, np.sqrt(1 - lam)]], dtype=complex),
            np.array([[0, 0], [0, np.sqrt(lam)]], dtype=complex)]


def t1_t2_channel_params(t: float, T1: float, T2: float):
    """Amplitude-damping gamma and phase-damping lambda for idling a time t.

    Requires T2 <= 2 T1. Coherences decay as exp(-t/T2) overall.
    """
    if T2 > 2 * T1:
        raise ValueError("T2 cannot exceed 2*T1")
    gamma = 1 - np.exp(-t / T1)
    inv_tphi = 1 / T2 - 1 / (2 * T1)
    lam = 1 - np.exp(-2 * t * inv_tphi)
    return float(gamma), float(lam)


def idle(rho: np.ndarray, t: float, T1: float, T2: float, qubit: int, n: int) -> np.ndarray:
    gamma, lam = t1_t2_channel_params(t, T1, T2)
    rho = apply_kraus(rho, amplitude_damping(gamma), qubit, n)
    return apply_kraus(rho, phase_damping(lam), qubit, n)


def noisy_bell_fidelity(p: float) -> float:
    """Fidelity with |Phi+> of a Bell pair when each qubit suffers depolarising
    noise of strength p after the H and CNOT gates."""
    from .circuit import bell_circuit
    from .qubit import bell_state, density_matrix

    rho = density_matrix(bell_circuit().state())
    for q in (0, 1):
        rho = apply_kraus(rho, depolarizing(p), q, 2)
    target = bell_state(0)
    return float(np.vdot(target, rho @ target).real)


# ------------------------------------------------------------- mitigation
def zne_extrapolate(scales, values, order: int = 1) -> float:
    """Zero-noise extrapolation: fit value(scale) and evaluate at scale 0."""
    coeffs = np.polyfit(scales, values, order)
    return float(np.polyval(coeffs, 0.0))


def confusion_matrix(p01: float, p10: float) -> np.ndarray:
    """Single-qubit readout confusion: column j = true j, row i = read i."""
    return np.array([[1 - p01, p10], [p01, 1 - p10]])


def readout_mitigate(measured: np.ndarray, confusion: np.ndarray) -> np.ndarray:
    """Invert the confusion matrix, then clip and renormalise."""
    est = np.linalg.solve(confusion, measured)
    est = np.clip(est, 0, None)
    return est / est.sum()


# -------------------------------------------------------------- benchmarking
def rb_model(m, A, p, B):
    return A * p ** m + B


def rb_fit(lengths, survival):
    """Fit F(m) = A p^m + B; return (p, average error per Clifford for one qubit)."""
    popt, _ = curve_fit(rb_model, lengths, survival, p0=[0.5, 0.99, 0.5], maxfev=10000)
    p = float(popt[1])
    return p, (1 - p) / 2
