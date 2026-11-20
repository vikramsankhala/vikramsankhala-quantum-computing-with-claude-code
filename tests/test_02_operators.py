import numpy as np
from qcwork import operators as op
from qcwork import hilbert as h


def test_paulis_are_hermitian_unitary_and_square_to_identity():
    for P in (op.X, op.Y, op.Z):
        assert op.is_hermitian(P) and op.is_unitary(P)
        assert np.allclose(P @ P, op.I2)


def test_pauli_commutation_relations():
    assert np.allclose(op.commutator(op.X, op.Y), 2j * op.Z)
    assert np.allclose(op.anticommutator(op.X, op.Z), 0)


def test_born_rule_probabilities_sum_to_one():
    psi = h.normalize(h.ket(1, 2j))
    probs = op.born_probabilities(psi, op.Z)
    assert np.isclose(sum(p for _, p in probs), 1.0)
    assert np.isclose(dict(probs)[1.0], 1 / 5)


def test_measurement_collapses_and_repeats():
    rng = np.random.default_rng(3)
    psi = h.normalize(h.ket(1, 1))
    val, post = op.measure(psi, op.Z, rng)
    val2, _ = op.measure(post, op.Z, rng)
    assert val == val2


def test_measurement_statistics_match_born_rule():
    rng = np.random.default_rng(5)
    psi = h.normalize(h.ket(1, 1j * np.sqrt(3)))  # P(0)=1/4
    outcomes = [op.measure(psi, op.Z, rng)[0] for _ in range(4000)]
    assert abs(np.mean(np.array(outcomes) == 1.0) - 0.25) < 0.03


def test_schrodinger_exact_matches_rk4():
    H = 0.7 * op.X + 0.3 * op.Z
    psi0 = h.ket(1, 0)
    exact = op.evolve(psi0, H, 2.0)
    num = op.schrodinger_rk4(psi0, H, 2.0, steps=2000)
    assert np.allclose(exact, num, atol=1e-9)


def test_time_evolution_is_unitary_and_preserves_norm():
    H = op.pauli_string("XX") + 0.5 * op.pauli_string("ZI")
    U = op.time_evolution_unitary(H, 1.3)
    assert op.is_unitary(U)
    psi = h.normalize(np.arange(1, 5) + 1j)
    assert np.isclose(h.norm(U @ psi), 1.0)


def test_rabi_oscillation_under_sigma_x_hamiltonian():
    # H = (Omega/2) X flips |0> to |1> at t = pi / Omega
    Omega = 2.0
    psi = op.evolve(h.ket(1, 0), 0.5 * Omega * op.X, np.pi / Omega)
    assert np.isclose(abs(psi[1]) ** 2, 1.0)


def test_energy_is_conserved_under_evolution():
    H = op.pauli_string("XZ") + op.pauli_string("YY") + 0.3 * op.pauli_string("ZI")
    psi0 = h.normalize(np.array([1, 2j, -1, 0.5]))
    e0 = op.expectation(psi0, H)
    assert np.isclose(op.expectation(op.evolve(psi0, H, 3.7), H), e0)


def test_uncertainty_principle():
    rng = np.random.default_rng(1)
    for _ in range(20):
        psi = h.normalize(rng.normal(size=2) + 1j * rng.normal(size=2))
        lhs, rhs = op.uncertainty_product(psi, op.X, op.Y)
        assert lhs >= rhs - 1e-12


def test_spectral_decomposition_reconstructs_operator():
    A = op.pauli_string("XZ") + 2 * op.pauli_string("II")
    groups = op.spectral_decomposition(A)
    assert np.allclose(sum(l * P for l, P in groups), A)
    assert np.allclose(sum(P for _, P in groups), np.eye(4))
