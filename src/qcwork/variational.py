"""Module 12: variational algorithms (VQE and QAOA) and QUBO tools."""
from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from .circuit import Circuit
from .operators import pauli_string
from . import gates as G

# Two-qubit reduced Hamiltonian of H2 at 0.735 angstrom (electronic part, hartree),
# in the parity mapping with two-qubit reduction. Coefficients as published in
# the standard Qiskit/PennyLane chemistry tutorials.
H2_TERMS = {
    "II": -1.052373245772859,
    "ZI": 0.39793742484318045,
    "IZ": -0.39793742484318045,
    "ZZ": -0.01128010425623538,
    "XX": 0.18093119978423156,
}
H2_NUCLEAR_REPULSION = 0.7199689944489797


def hamiltonian_from_terms(terms: dict[str, float]) -> np.ndarray:
    return sum(c * pauli_string(p) for p, c in terms.items())


def h2_hamiltonian() -> np.ndarray:
    return hamiltonian_from_terms(H2_TERMS)


def h2_ansatz(theta: float) -> Circuit:
    """cos(theta/2)|01> + sin(theta/2)|10>: one parameter, two entangling gates."""
    return Circuit(2).ry(theta, 0).cx(0, 1).x(1)


def energy(terms: dict[str, float], circuit: Circuit) -> float:
    return sum(c * circuit.expectation(p) for p, c in terms.items())


def vqe_h2():
    """Minimise <H> over the one-parameter ansatz; returns (energy, theta)."""
    f = lambda th: energy(H2_TERMS, h2_ansatz(th))
    best = min((minimize_scalar(f, bounds=(a, a + np.pi), method="bounded") for a in np.linspace(-np.pi, np.pi, 4, endpoint=False)),
               key=lambda r: r.fun)
    return float(best.fun), float(best.x)


def parameter_shift_gradient(f, theta: float) -> float:
    """Exact gradient of f(theta) for a gate exp(-i theta P / 2): [f(t+pi/2) - f(t-pi/2)] / 2."""
    return 0.5 * (f(theta + np.pi / 2) - f(theta - np.pi / 2))


# ---------------------------------------------------------------- QAOA
def maxcut_cost_vector(n: int, edges) -> np.ndarray:
    """Cut value of every bitstring (index = bitstring with qubit 0 leftmost)."""
    costs = np.zeros(2 ** n)
    for x in range(2 ** n):
        bits = [(x >> (n - 1 - q)) & 1 for q in range(n)]
        costs[x] = sum(bits[i] != bits[j] for i, j in edges)
    return costs


def qaoa_state(n: int, edges, gammas, betas) -> np.ndarray:
    costs = maxcut_cost_vector(n, edges)
    c = Circuit(n)
    for q in range(n):
        c.h(q)
    for g, b in zip(gammas, betas):
        c.apply(np.diag(np.exp(-1j * g * costs)), list(range(n)), "cost")
        for q in range(n):
            c.rx(2 * b, q)
    return c.state()


def qaoa_expectation(n: int, edges, gammas, betas) -> float:
    psi = qaoa_state(n, edges, gammas, betas)
    return float((np.abs(psi) ** 2 * maxcut_cost_vector(n, edges)).sum())


def optimise_qaoa(n: int, edges, p: int = 1, restarts: int = 12, seed: int = 0):
    """Maximise the expected cut over 2p angles; returns (value, gammas, betas)."""
    rng = np.random.default_rng(seed)
    f = lambda v: -qaoa_expectation(n, edges, v[:p], v[p:])
    best = None
    for _ in range(restarts):
        res = minimize(f, rng.uniform(0, np.pi, 2 * p), method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-9})
        if best is None or res.fun < best.fun:
            best = res
    return float(-best.fun), best.x[:p], best.x[p:]


# ---------------------------------------------------------------- QUBO
def qubo_energy(Q: np.ndarray, x) -> float:
    x = np.asarray(x, dtype=float)
    return float(x @ Q @ x)


def brute_force_qubo(Q: np.ndarray):
    n = Q.shape[0]
    best = min(itertools.product((0, 1), repeat=n), key=lambda x: qubo_energy(Q, x))
    return best, qubo_energy(Q, best)


def qubo_to_ising(Q: np.ndarray):
    """x = (1 - z)/2 maps a QUBO to an Ising model: E = offset + sum h_i z_i + sum J_ij z_i z_j."""
    n = Q.shape[0]
    h = np.zeros(n)
    J = np.zeros((n, n))
    offset = 0.0
    for i in range(n):
        offset += Q[i, i] / 2
        h[i] -= Q[i, i] / 2
        for j in range(i + 1, n):
            q = Q[i, j] + Q[j, i]
            offset += q / 4
            h[i] -= q / 4
            h[j] -= q / 4
            J[i, j] += q / 4
    return h, J, offset


def ising_energy(h, J, offset, z) -> float:
    z = np.asarray(z, dtype=float)
    return float(offset + h @ z + z @ J @ z)


def portfolio_qubo(mu, sigma, risk_aversion: float, k: int, penalty: float) -> np.ndarray:
    """Choose exactly k assets: minimise risk*x'Sx - mu'x + penalty (sum x - k)^2."""
    mu = np.asarray(mu, dtype=float)
    n = len(mu)
    Q = risk_aversion * np.asarray(sigma, dtype=float) + penalty * (np.ones((n, n)) - 2 * k * np.eye(n))
    Q = Q - np.diag(mu)
    return Q
