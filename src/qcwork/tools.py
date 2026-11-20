"""Tool functions exposed to AI coding agents (Claude Code, Codex CLI, Gemini CLI)
through the MCP server in mcp_server/server.py. They take and return plain JSON
types so any agent can call them."""
from __future__ import annotations

import numpy as np

from .circuit import Circuit
from . import contextuality, qec
from .qubit import bloch_vector, state_from_bloch

_ONE = {"h", "x", "y", "z", "s", "sdg", "t"}
_ONE_ANGLE = {"rx", "ry", "rz", "p"}
_TWO = {"cx", "cz", "swap"}


def run_circuit(n: int, gates: list[dict], shots: int = 0, seed: int = 0) -> dict:
    """Run a circuit given as [{"gate": "h", "qubits": [0]}, {"gate": "rx", "angle": 0.5, "qubits": [1]}, ...]."""
    c = Circuit(n)
    for g in gates:
        name, qs = g["gate"].lower(), g["qubits"]
        if name in _ONE | _TWO | {"ccx"}:
            getattr(c, name)(*qs)
        elif name in _ONE_ANGLE:
            getattr(c, name)(float(g["angle"]), *qs)
        else:
            raise ValueError(f"unknown gate {name!r}")
    probs = c.probabilities()
    out = {
        "probabilities": {format(i, f"0{n}b"): round(float(p), 10) for i, p in enumerate(probs) if p > 1e-12},
        "depth": c.depth(),
        "gate_count": c.gate_count(),
        "qasm": c.to_qasm(),
    }
    if shots:
        out["counts"] = c.sample(shots, seed)
    return out


def bloch(theta: float, phi: float) -> dict:
    v = bloch_vector(state_from_bloch(theta, phi))
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def estimate_resources(logical_qubits: int, physical_error_rate: float, target_failure: float = 0.01,
                       rounds: float = 1e6, overhead: float = 1.5) -> dict:
    """Back-of-envelope surface-code estimate (a teaching model, not a vendor estimator)."""
    d = qec.distance_for_target(physical_error_rate, target_failure, logical_qubits, rounds)
    return {"code_distance": d, "physical_qubits": qec.physical_qubits(logical_qubits, d, overhead),
            "assumptions": "A=0.1, threshold=1%, overhead factor applied to data+ancilla qubits"}


def magic_square() -> dict:
    return {"classical_value": contextuality.classical_value(),
            "quantum_value": contextuality.quantum_value(),
            "ks_bell_classical_bound": contextuality.ks_bell_classical_bound(),
            "ks_bell_quantum_value": 6.0}
