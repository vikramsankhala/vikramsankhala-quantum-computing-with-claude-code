import numpy as np
import pytest
from qcwork import algorithms as alg


def test_deutsch_jozsa_distinguishes_constant_from_balanced_in_one_query():
    n = 3
    assert alg.deutsch_jozsa(lambda x: 0, n)[0] == "constant"
    assert alg.deutsch_jozsa(lambda x: 1, n)[0] == "constant"
    assert alg.deutsch_jozsa(lambda x: x & 1, n)[0] == "balanced"
    assert alg.deutsch_jozsa(lambda x: bin(x).count("1") % 2, n)[0] == "balanced"
    assert np.isclose(alg.deutsch_jozsa(lambda x: 0, n)[1], 1.0)
    assert np.isclose(alg.deutsch_jozsa(lambda x: x & 1, n)[1], 0.0)


@pytest.mark.parametrize("secret", ["1", "101", "0110", "11111"])
def test_bernstein_vazirani_recovers_secret(secret):
    assert alg.bernstein_vazirani(secret) == secret


@pytest.mark.parametrize("n,marked", [(3, {5}), (4, {9}), (5, {3, 17}), (6, {42})])
def test_grover_matches_closed_form(n, marked):
    N = 2 ** n
    probs = alg.grover(n, marked)
    k = alg.grover_optimal_iterations(N, len(marked))
    expected = alg.grover_success_probability(N, len(marked), k)
    assert np.isclose(sum(probs[m] for m in marked), expected)
    assert expected > 0.9


def test_grover_overshoot_reduces_success():
    N = 64
    k = alg.grover_optimal_iterations(N)
    assert alg.grover_success_probability(N, 1, 2 * k) < alg.grover_success_probability(N, 1, k)


def test_grover_iterations_scale_as_sqrt_n():
    assert alg.grover_optimal_iterations(2 ** 20) == 804  # floor(pi/4 * 1024)


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_qft_circuit_equals_dft_matrix(n):
    assert np.allclose(alg.qft_circuit(n).unitary(), alg.qft_matrix(n))


def test_qft_matrix_is_unitary():
    F = alg.qft_matrix(4)
    assert np.allclose(F.conj().T @ F, np.eye(16))


def test_phase_estimation_exact_phase():
    U = np.diag([1, np.exp(2j * np.pi * 0.25)])
    probs = alg.phase_estimation(U, np.array([0, 1]), t=3)
    assert np.isclose(probs[2], 1.0)


def test_phase_estimation_inexact_phase_meets_textbook_bound():
    phi = 1 / 3
    U = np.diag([1, np.exp(2j * np.pi * phi)])
    t = 8
    probs = alg.phase_estimation(U, np.array([0, 1]), t)
    x = int(np.floor(phi * 2 ** t))
    assert probs[x] + probs[x + 1] >= 8 / np.pi ** 2 - 1e-9
    assert abs(np.argmax(probs) / 2 ** t - phi) <= 1 / 2 ** t


def test_multiplicative_order():
    assert alg.multiplicative_order(7, 15) == 4
    assert alg.multiplicative_order(2, 21) == 6


def test_order_finding_distribution_peaks_at_multiples_of_T_over_r():
    t = 8
    dist = alg.order_finding_distribution(7, 15, t)
    peaks = np.argsort(dist)[-4:]
    assert set(peaks) == {0, 64, 128, 192}
    assert np.isclose(dist[peaks].sum(), 1.0)


@pytest.mark.parametrize("N,a,expected", [(15, 7, {3, 5}), (21, 2, {3, 7}), (35, 3, {5, 7})])
def test_shor_factors_small_numbers(N, a, expected):
    res = alg.shor_factor(N, a, seed=11)
    assert res is not None and set(res) == expected
