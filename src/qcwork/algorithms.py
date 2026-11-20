"""Modules 5 and 6: oracle algorithms, Grover search, the QFT, phase
estimation and Shor's order finding."""
from __future__ import annotations

from fractions import Fraction
from math import asin, floor, gcd, pi, sin, sqrt

import numpy as np

from .circuit import Circuit
from . import gates as G


# ---------------------------------------------------------------- oracles
def oracle_matrix(f, n: int) -> np.ndarray:
    """U_f |x>|y> = |x>|y xor f(x)> on n input qubits and one ancilla."""
    N = 2 ** (n + 1)
    U = np.zeros((N, N), dtype=complex)
    for x in range(2 ** n):
        for y in (0, 1):
            U[(x << 1) | (y ^ f(x)), (x << 1) | y] = 1
    return U


def deutsch_jozsa(f, n: int):
    """Decide whether f is constant or balanced with one query.

    Returns (verdict, probability that the input register reads all zeros).
    """
    c = Circuit(n + 1).x(n)
    for q in range(n + 1):
        c.h(q)
    c.apply(oracle_matrix(f, n), list(range(n + 1)), "Uf")
    for q in range(n):
        c.h(q)
    probs = c.probabilities().reshape([2] * n + [2])
    p_zero = float(probs[(0,) * n].sum())
    return ("constant" if p_zero > 0.5 else "balanced"), p_zero


def bernstein_vazirani(secret: str) -> str:
    """Recover the hidden string s from f(x) = s.x mod 2 with one query."""
    n = len(secret)
    s = int(secret, 2)
    f = lambda x: bin(x & s).count("1") % 2
    c = Circuit(n + 1).x(n)
    for q in range(n + 1):
        c.h(q)
    c.apply(oracle_matrix(f, n), list(range(n + 1)), "Uf")
    for q in range(n):
        c.h(q)
    probs = c.probabilities().reshape(2 ** n, 2).sum(axis=1)
    return format(int(np.argmax(probs)), f"0{n}b")


# ------------------------------------------------------------------ Grover
def grover_theta(N: int, M: int = 1) -> float:
    return asin(sqrt(M / N))


def grover_success_probability(N: int, M: int, k: int) -> float:
    """sin^2((2k+1) theta), the success probability after k iterations."""
    return sin((2 * k + 1) * grover_theta(N, M)) ** 2


def grover_optimal_iterations(N: int, M: int = 1) -> int:
    return floor(pi / (4 * grover_theta(N, M)))


def grover(n: int, marked, iterations: int | None = None) -> np.ndarray:
    """Run Grover search on n qubits; return the final probability vector."""
    marked = set(marked)
    N = 2 ** n
    k = grover_optimal_iterations(N, len(marked)) if iterations is None else iterations
    oracle = np.diag([-1 if x in marked else 1 for x in range(N)]).astype(complex)
    zero_flip = np.diag([-1] + [1] * (N - 1)).astype(complex)
    c = Circuit(n)
    for q in range(n):
        c.h(q)
    for _ in range(k):
        c.apply(oracle, list(range(n)), "oracle")
        for q in range(n):
            c.h(q)
        c.apply(zero_flip, list(range(n)), "2|0><0|-I")
        for q in range(n):
            c.h(q)
    return c.probabilities()


# --------------------------------------------------------------------- QFT
def qft_matrix(n: int) -> np.ndarray:
    N = 2 ** n
    w = np.exp(2j * pi / N)
    j, k = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    return w ** (j * k) / sqrt(N)


def qft_circuit(n: int) -> Circuit:
    """The textbook QFT: Hadamards, controlled phases and a final swap layer."""
    c = Circuit(n)
    for j in range(n):
        c.h(j)
        for k in range(j + 1, n):
            c.cp(pi / 2 ** (k - j), k, j)
    for j in range(n // 2):
        c.swap(j, n - 1 - j)
    return c


# ------------------------------------------------------- phase estimation
def phase_estimation(U: np.ndarray, eigvec: np.ndarray, t: int) -> np.ndarray:
    """Probability distribution over t-bit estimates of the eigenphase of U.

    U |v> = exp(2 pi i phi) |v>; the outcome x estimates phi ~ x / 2^t.
    """
    T = 2 ** t
    joint = np.zeros((T, len(eigvec)), dtype=complex)
    v = eigvec.astype(complex)
    for x in range(T):
        joint[x] = v / sqrt(T)
        v = U @ v
    out = np.conj(qft_matrix(t)) @ joint  # inverse QFT on the counting register
    return (np.abs(out) ** 2).sum(axis=1)


# -------------------------------------------------------------------- Shor
def order_finding_distribution(a: int, N: int, t: int) -> np.ndarray:
    """Distribution of the counting register in Shor's order-finding circuit."""
    m = int(np.ceil(np.log2(N)))
    T = 2 ** t
    psi = np.zeros((T, 2 ** m), dtype=complex)
    for x in range(T):
        psi[x, pow(a, x, N)] = 1 / sqrt(T)
    out = np.fft.fft(psi, axis=0) / sqrt(T)
    return (np.abs(out) ** 2).sum(axis=1)


def multiplicative_order(a: int, N: int) -> int:
    r, v = 1, a % N
    while v != 1:
        v = (v * a) % N
        r += 1
    return r


def recover_order(y: int, t: int, a: int, N: int) -> int | None:
    """Continued-fraction step: turn a measured y into a candidate order."""
    frac = Fraction(y, 2 ** t).limit_denominator(N)
    r = frac.denominator
    return r if r > 0 and pow(a, r, N) == 1 else None


def shor_factor(N: int, a: int, t: int | None = None, seed: int = 0, attempts: int = 20):
    """Factor N using simulated quantum order finding for the base a.

    Returns (p, q) or None. The quantum step is simulated exactly; the rest is
    the classical post-processing of Shor's algorithm.
    """
    if gcd(a, N) != 1:
        d = gcd(a, N)
        return d, N // d
    m = int(np.ceil(np.log2(N)))
    t = 2 * m if t is None else t
    dist = order_finding_distribution(a, N, t)
    rng = np.random.default_rng(seed)
    for _ in range(attempts):
        y = int(rng.choice(len(dist), p=dist / dist.sum()))
        r = recover_order(y, t, a, N)
        if r and r % 2 == 0:
            x = pow(a, r // 2, N)
            if x != N - 1:
                p = gcd(x - 1, N)
                if 1 < p < N:
                    return p, N // p
    return None
