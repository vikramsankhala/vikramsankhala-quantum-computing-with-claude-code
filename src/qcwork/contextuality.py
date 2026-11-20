"""Module 11: contextuality and the Mermin-Peres magic square game.

Rules: the referee gives Alice a row r and Bob a column c. Alice answers three
bits with an EVEN number of 1s, Bob three bits with an ODD number of 1s. They
win when their bits agree on the shared cell (r, c).
"""
from __future__ import annotations

import itertools

import numpy as np

from .operators import pauli_string

# The Peres-Mermin square of two-qubit Pauli observables.
SQUARE = [["XI", "IX", "XX"], ["IZ", "ZI", "ZZ"], ["XZ", "ZX", "YY"]]
ROW_SIGNS = [1, 1, 1]
COL_SIGNS = [1, 1, -1]


def square_matrices():
    return [[pauli_string(s) for s in row] for row in SQUARE]


def line_product(ops) -> np.ndarray:
    return ops[0] @ ops[1] @ ops[2]


def lines_commute() -> bool:
    M = square_matrices()
    lines = [M[r] for r in range(3)] + [[M[r][c] for r in range(3)] for c in range(3)]
    for line in lines:
        for a, b in itertools.combinations(line, 2):
            if not np.allclose(a @ b, b @ a):
                return False
    return True


def line_products():
    """Return (row_products, col_products) as scalar multiples of the identity."""
    M = square_matrices()
    out = []
    for lines in ([M[r] for r in range(3)], [[M[r][c] for r in range(3)] for c in range(3)]):
        res = []
        for line in lines:
            P = line_product(line)
            s = P[0, 0].real
            assert np.allclose(P, s * np.eye(4))
            res.append(int(round(s)))
        out.append(res)
    return out[0], out[1]


# ------------------------------------------------------- the classical game
def _lines_with_parity(parity: int):
    return [t for t in itertools.product((0, 1), repeat=3) if sum(t) % 2 == parity]


def classical_value() -> float:
    """Best win probability over all deterministic classical strategies."""
    rows = _lines_with_parity(0)  # Alice: even
    cols = _lines_with_parity(1)  # Bob: odd
    best = 0
    for alice in itertools.product(rows, repeat=3):
        for bob in itertools.product(cols, repeat=3):
            wins = sum(alice[r][c] == bob[c][r] for r in range(3) for c in range(3))
            best = max(best, wins)
    return best / 9


def has_classical_grid(n: int) -> bool:
    """Is there an n x n bit grid with all rows even and all columns odd?

    Solved as a linear system over GF(2): True exactly when n is even.
    """
    eqs = []
    for r in range(n):
        eqs.append(([r * n + c for c in range(n)], 0))
    for c in range(n):
        eqs.append(([r * n + c for r in range(n)], 1))
    A = np.zeros((len(eqs), n * n + 1), dtype=np.uint8)
    for i, (idx, rhs) in enumerate(eqs):
        A[i, idx] = 1
        A[i, -1] = rhs
    row = 0
    for col in range(n * n):
        pivot = next((i for i in range(row, len(eqs)) if A[i, col]), None)
        if pivot is None:
            continue
        A[[row, pivot]] = A[[pivot, row]]
        for i in range(len(eqs)):
            if i != row and A[i, col]:
                A[i] ^= A[row]
        row += 1
    return not any((A[i, :-1] == 0).all() and A[i, -1] == 1 for i in range(len(eqs)))


def classical_bound_value(n: int) -> float:
    """Best classical win probability for odd n: (n^2 - 1) / n^2 (proved in the book)."""
    return 1.0 if n % 2 == 0 else (n * n - 1) / (n * n)


# -------------------------------------------------------- the quantum game
def two_bell_pairs() -> np.ndarray:
    """|Phi+>_{a1 b1} |Phi+>_{a2 b2} written in qubit order (a1, a2, b1, b2)."""
    s = np.zeros((2, 2, 2, 2), dtype=complex)
    for i in (0, 1):
        for j in (0, 1):
            s[i, j, i, j] = 0.5
    return s.reshape(16)


def quantum_win_probability(r: int, c: int) -> float:
    """Exact winning probability for the questions (r, c) with the Pauli strategy."""
    M = square_matrices()
    psi = two_bell_pairs()
    I4 = np.eye(4)
    alice_ops = [np.kron(M[r][k], I4) for k in range(3)]
    # Bell pairs transpose observables from one side to the other.
    bob_ops = [np.kron(I4, M[k][c].T) for k in range(3)]
    win = 0.0
    for sa in itertools.product((1, -1), repeat=3):
        if np.prod(sa) != ROW_SIGNS[r]:
            continue
        Pa = np.eye(16)
        for s, O in zip(sa, alice_ops):
            Pa = Pa @ ((np.eye(16) + s * O) / 2)
        for sb in itertools.product((1, -1), repeat=3):
            if np.prod(sb) != COL_SIGNS[c]:
                continue
            if sa[c] != sb[r]:
                continue
            Pb = np.eye(16)
            for s, O in zip(sb, bob_ops):
                Pb = Pb @ ((np.eye(16) + s * O) / 2)
            win += float(np.vdot(psi, Pa @ Pb @ psi).real)
    return win


def quantum_value() -> float:
    """Average winning probability over the nine equally likely questions."""
    return float(np.mean([quantum_win_probability(r, c) for r in range(3) for c in range(3)]))


# ------------------------------------------------ state-independent inequality
def ks_bell_classical_bound() -> int:
    """Maximum of R1+R2+R3+C1+C2-C3 over all +-1 assignments of the nine cells."""
    best = -99
    for v in itertools.product((1, -1), repeat=9):
        g = np.array(v).reshape(3, 3)
        rows = g.prod(axis=1)
        cols = g.prod(axis=0)
        best = max(best, int(rows.sum() + cols[0] + cols[1] - cols[2]))
    return best


def ks_bell_quantum_value(psi: np.ndarray) -> float:
    """The same sum evaluated quantum mechanically; equals 6 for every state."""
    M = square_matrices()
    total = 0.0
    for r in range(3):
        total += ROW_SIGNS[r] * 0 + float(np.vdot(psi, line_product(M[r]) @ psi).real)
    for c in range(3):
        col = [M[r][c] for r in range(3)]
        sign = 1 if c < 2 else -1
        total += sign * float(np.vdot(psi, line_product(col) @ psi).real)
    return total
