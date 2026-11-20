import numpy as np
from qcwork import hilbert as h


def test_inner_product_is_conjugate_linear_in_first_argument():
    a = h.ket(1, 1j)
    b = h.ket(2, 1)
    assert np.isclose(h.inner(1j * a, b), -1j * h.inner(a, b))


def test_norm_and_normalize():
    v = h.ket(3, 4j)
    assert np.isclose(h.norm(v), 5)
    assert h.is_normalized(h.normalize(v))


def test_cauchy_schwarz_holds_for_random_vectors():
    rng = np.random.default_rng(0)
    for _ in range(50):
        a = rng.normal(size=4) + 1j * rng.normal(size=4)
        b = rng.normal(size=4) + 1j * rng.normal(size=4)
        assert h.cauchy_schwarz_gap(a, b) >= -1e-12


def test_gram_schmidt_gives_orthonormal_basis_and_drops_dependents():
    vs = [h.ket(1, 1, 0), h.ket(1, 0, 1), h.ket(2, 1, 1), h.ket(0, 1, 1)]
    basis = h.gram_schmidt(vs)
    assert len(basis) == 3
    assert h.is_orthonormal(basis)


def test_expansion_in_a_basis_reconstructs_the_state():
    basis = h.gram_schmidt([h.ket(1, 1j, 0), h.ket(0, 1, 1), h.ket(1, 0, 2)])
    psi = h.normalize(h.ket(0.3, -0.2j, 0.9))
    c = h.coefficients(psi, basis)
    assert np.allclose(h.reconstruct(c, basis), psi)
    assert np.isclose((abs(c) ** 2).sum(), 1.0)  # Parseval


def test_dimension_grows_exponentially_with_qubits():
    assert [h.dimension_of_n_qubits(n) for n in (1, 2, 10, 20)] == [2, 4, 1024, 1048576]


def test_projector_is_idempotent_and_hermitian():
    P = h.projector(h.ket(1, 1j))
    assert np.allclose(P @ P, P) and np.allclose(P, P.conj().T)
