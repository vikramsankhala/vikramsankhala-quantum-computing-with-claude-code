"""Module 16: BB84 key distribution and post-quantum security arithmetic."""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def binary_entropy(x: float) -> float:
    if x <= 0 or x >= 1:
        return 0.0
    return float(-x * np.log2(x) - (1 - x) * np.log2(1 - x))


def asymptotic_key_fraction(qber: float) -> float:
    """BB84 asymptotic secret fraction of the sifted key: max(0, 1 - 2 h(e))."""
    return max(0.0, 1 - 2 * binary_entropy(qber))


def qber_threshold() -> float:
    """The error rate at which the asymptotic BB84 key rate reaches zero (about 11%)."""
    return float(brentq(lambda e: 1 - 2 * binary_entropy(e), 0.01, 0.3))


def bb84(n: int, eve: bool = False, channel_error: float = 0.0, seed: int = 0) -> dict:
    """Simulate n BB84 rounds; Eve (optional) intercepts and resends in a random basis."""
    rng = np.random.default_rng(seed)
    a_bits = rng.integers(0, 2, n)
    a_basis = rng.integers(0, 2, n)
    b_basis = rng.integers(0, 2, n)
    bits, basis = a_bits, a_basis
    if eve:
        e_basis = rng.integers(0, 2, n)
        e_bits = np.where(e_basis == a_basis, a_bits, rng.integers(0, 2, n))
        bits, basis = e_bits, e_basis
    b_bits = np.where(b_basis == basis, bits, rng.integers(0, 2, n))
    b_bits = b_bits ^ (rng.random(n) < channel_error)
    sift = a_basis == b_basis
    qber = float(np.mean(a_bits[sift] != b_bits[sift]))
    return {"sifted": int(sift.sum()), "qber": qber, "key_fraction": asymptotic_key_fraction(qber)}


def grover_effective_security_bits(key_bits: int) -> float:
    """A k-bit symmetric key offers about k/2 bits against Grover search."""
    return key_bits / 2


def grover_aes_queries(key_bits: int) -> float:
    """Approximate Grover iterations (pi/4) 2^(k/2)."""
    return float(np.pi / 4 * 2 ** (key_bits / 2))
