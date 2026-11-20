"""Module 4: the circuit model, a from-scratch state-vector simulator.

Qubit 0 is the leftmost (most significant) qubit. The simulator stores the
state as an array with one axis per qubit, so a k-qubit gate is a tensor
contraction over k axes.
"""
from __future__ import annotations

import numpy as np

from . import gates as G


class Circuit:
    def __init__(self, n: int, state: np.ndarray | None = None):
        self.n = n
        if state is None:
            self._psi = np.zeros(2 ** n, dtype=complex)
            self._psi[0] = 1.0
        else:
            self._psi = np.array(state, dtype=complex).reshape(-1)
        self.ops: list[tuple[str, tuple[int, ...], tuple, np.ndarray]] = []

    # -- core ---------------------------------------------------------
    def apply(self, U: np.ndarray, qubits, name: str = "U", params: tuple = ()) -> "Circuit":
        qubits = tuple(int(q) for q in qubits)
        k = len(qubits)
        Ut = np.asarray(U, dtype=complex).reshape([2] * (2 * k))
        psi = self._psi.reshape([2] * self.n)
        psi = np.tensordot(Ut, psi, axes=(list(range(k, 2 * k)), list(qubits)))
        psi = np.moveaxis(psi, list(range(k)), list(qubits))
        self._psi = psi.reshape(-1)
        self.ops.append((name, qubits, tuple(params), np.asarray(U, dtype=complex)))
        return self

    # -- named gates --------------------------------------------------
    def h(self, q): return self.apply(G.H, [q], "h")
    def x(self, q): return self.apply(G.X, [q], "x")
    def y(self, q): return self.apply(G.Y, [q], "y")
    def z(self, q): return self.apply(G.Z, [q], "z")
    def s(self, q): return self.apply(G.S, [q], "s")
    def sdg(self, q): return self.apply(G.SDG, [q], "sdg")
    def t(self, q): return self.apply(G.T, [q], "t")
    def rx(self, theta, q): return self.apply(G.rx(theta), [q], "rx", (theta,))
    def ry(self, theta, q): return self.apply(G.ry(theta), [q], "ry", (theta,))
    def rz(self, theta, q): return self.apply(G.rz(theta), [q], "rz", (theta,))
    def p(self, phi, q): return self.apply(G.phase(phi), [q], "p", (phi,))
    def cx(self, c, t): return self.apply(G.CNOT, [c, t], "cx")
    def cz(self, c, t): return self.apply(G.CZ, [c, t], "cz")
    def cp(self, phi, c, t): return self.apply(G.cphase(phi), [c, t], "cp", (phi,))
    def swap(self, a, b): return self.apply(G.SWAP, [a, b], "swap")
    def ccx(self, c1, c2, t): return self.apply(G.TOFFOLI, [c1, c2, t], "ccx")

    # -- state access -------------------------------------------------
    def state(self) -> np.ndarray:
        return self._psi.copy()

    def probabilities(self) -> np.ndarray:
        return np.abs(self._psi) ** 2

    def copy(self) -> "Circuit":
        c = Circuit(self.n, self._psi)
        c.ops = list(self.ops)
        return c

    def sample(self, shots: int, seed: int | None = None) -> dict[str, int]:
        rng = np.random.default_rng(seed)
        p = self.probabilities()
        draws = rng.choice(len(p), size=shots, p=p / p.sum())
        counts: dict[str, int] = {}
        for d in draws:
            key = format(int(d), f"0{self.n}b")
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    def measure(self, q: int, outcome: int | None = None, rng: np.random.Generator | None = None) -> int:
        """Projective Z measurement of qubit q; collapses the state.

        Pass `outcome` to force a branch (used to check every branch of a
        protocol such as teleportation)."""
        psi = self._psi.reshape([2] * self.n)
        p0 = float(np.sum(np.abs(np.take(psi, 0, axis=q)) ** 2))
        if outcome is None:
            rng = rng or np.random.default_rng()
            outcome = 0 if rng.random() < p0 else 1
        p = p0 if outcome == 0 else 1 - p0
        if p < 1e-14:
            raise ValueError("forced outcome has zero probability")
        keep = np.zeros_like(psi)
        idx = [slice(None)] * self.n
        idx[q] = outcome
        keep[tuple(idx)] = psi[tuple(idx)]
        self._psi = (keep / np.sqrt(p)).reshape(-1)
        return int(outcome)

    def expectation(self, pauli: str) -> float:
        """<psi| P |psi> for a Pauli string such as 'XZI'."""
        from .operators import PAULIS

        tmp = Circuit(self.n, self._psi)
        for q, ch in enumerate(pauli):
            if ch != "I":
                tmp.apply(PAULIS[ch], [q])
        return float(np.vdot(self._psi, tmp._psi).real)

    # -- circuit-level views -----------------------------------------
    def unitary(self) -> np.ndarray:
        """The full 2^n x 2^n unitary of the recorded gates (small n only)."""
        d = 2 ** self.n
        cols = []
        for j in range(d):
            e = np.zeros(d, dtype=complex)
            e[j] = 1
            c = Circuit(self.n, e)
            for _, qs, _, U in self.ops:
                c.apply(U, qs)
            cols.append(c._psi)
        return np.column_stack(cols)

    def depth(self) -> int:
        level = [0] * self.n
        for _, qs, _, _ in self.ops:
            m = max(level[q] for q in qs) + 1
            for q in qs:
                level[q] = m
        return max(level) if level else 0

    def gate_count(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for name, _, _, _ in self.ops:
            out[name] = out.get(name, 0) + 1
        return out

    def to_qasm(self) -> str:
        """Minimal OpenQASM 2.0 export for named gates."""
        lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{self.n}];"]
        for name, qs, params, _ in self.ops:
            args = ",".join(f"q[{q}]" for q in qs)
            if name in {"rx", "ry", "rz"}:
                lines.append(f"{name}({params[0]}) {args};")
            elif name == "p":
                lines.append(f"u1({params[0]}) {args};")
            elif name == "cp":
                lines.append(f"cu1({params[0]}) {args};")
            elif name in {"h", "x", "y", "z", "s", "sdg", "t", "cx", "cz", "swap", "ccx"}:
                lines.append(f"{name} {args};")
            else:
                raise ValueError(f"cannot export gate {name!r} to QASM")
        return "\n".join(lines) + "\n"


def bell_circuit() -> Circuit:
    return Circuit(2).h(0).cx(0, 1)


def ghz_circuit(n: int) -> Circuit:
    c = Circuit(n).h(0)
    for k in range(n - 1):
        c.cx(k, k + 1)
    return c
