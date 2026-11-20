import numpy as np
from qcwork import variational as v

EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]


def test_h2_vqe_matches_exact():
    e, _ = v.vqe_h2()
    assert abs(e - np.linalg.eigvalsh(v.h2_hamiltonian())[0]) < 1e-6


def test_parameter_shift_matches_finite_difference():
    f = lambda t: v.energy(v.H2_TERMS, v.h2_ansatz(t))
    t = 0.7
    fd = (f(t + 1e-6) - f(t - 1e-6)) / 2e-6
    assert abs(v.parameter_shift_gradient(f, t) - fd) < 1e-5


def test_qaoa_p1_and_p2():
    assert abs(v.optimise_qaoa(4, EDGES, 1)[0] - 3.0) < 1e-4
    assert v.optimise_qaoa(4, EDGES, 2)[0] > 3.99


def test_qubo_ising_equivalence():
    rng = np.random.default_rng(1)
    Q = rng.normal(size=(5, 5))
    h, J, off = v.qubo_to_ising(Q)
    for x in np.ndindex(*(2,) * 5):
        z = 1 - 2 * np.array(x)
        assert abs(v.qubo_energy(Q, x) - v.ising_energy(h, J, off, z)) < 1e-9


def test_portfolio_picks_k_assets():
    mu = [0.1, 0.2, 0.15, 0.05]
    S = np.diag([0.1, 0.2, 0.1, 0.05])
    x, _ = v.brute_force_qubo(v.portfolio_qubo(mu, S, 1.0, 2, 5.0))
    assert sum(x) == 2
