import numpy as np
from qcwork import qubit as q
from qcwork.hilbert import tensor, normalize


def test_bloch_vector_is_unit_length_for_pure_states():
    rng = np.random.default_rng(2)
    for _ in range(20):
        psi = normalize(rng.normal(size=2) + 1j * rng.normal(size=2))
        assert np.isclose(np.linalg.norm(q.bloch_vector(psi)), 1.0)


def test_bloch_parametrisation_round_trip():
    th, ph = 1.1, 0.7
    v = q.bloch_vector(q.state_from_bloch(th, ph))
    assert np.allclose(v, [np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])


def test_cardinal_states():
    assert np.allclose(q.bloch_vector(q.ZERO), [0, 0, 1])
    assert np.allclose(q.bloch_vector(q.PLUS), [1, 0, 0])


def test_purity_of_pure_and_maximally_mixed_states():
    assert np.isclose(q.purity(q.density_matrix(q.PLUS)), 1.0)
    assert np.isclose(q.purity(np.eye(2) / 2), 0.5)


def test_product_state_has_no_entanglement():
    psi = tensor(q.PLUS, q.ZERO)
    assert q.is_product_state(psi, 1, 2)
    assert np.isclose(q.entanglement_entropy(psi, 1, 2), 0.0)


def test_bell_states_are_maximally_entangled():
    for k in range(4):
        psi = q.bell_state(k)
        assert not q.is_product_state(psi, 1, 2)
        assert np.isclose(q.entanglement_entropy(psi, 1, 2), 1.0)
        rho_a = q.reduced_density_matrix(psi, [0], 2)
        assert np.allclose(rho_a, np.eye(2) / 2)


def test_ghz_reduced_state_is_classical_mixture():
    rho = q.reduced_density_matrix(q.ghz_state(3), [0], 3)
    assert np.allclose(rho, np.eye(2) / 2)
    assert np.isclose(q.entanglement_entropy(q.ghz_state(3), 1, 3), 1.0)
