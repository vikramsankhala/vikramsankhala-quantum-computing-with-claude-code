import numpy as np
import pytest
from qcwork import contextuality as cx
from qcwork.hilbert import normalize


def test_every_row_and_column_commutes():
    assert cx.lines_commute()


def test_line_products_are_plus_identity_except_last_column():
    rows, cols = cx.line_products()
    assert rows == [1, 1, 1]
    assert cols == [1, 1, -1]


def test_best_classical_strategy_wins_eight_ninths():
    assert np.isclose(cx.classical_value(), 8 / 9)


@pytest.mark.parametrize("n,expected", [(2, True), (3, False), (4, True), (5, False), (6, True), (7, False)])
def test_classical_grid_exists_only_for_even_n(n, expected):
    assert cx.has_classical_grid(n) is expected


def test_classical_bound_formula():
    assert np.isclose(cx.classical_bound_value(3), 8 / 9)
    assert np.isclose(cx.classical_bound_value(5), 24 / 25)
    assert cx.classical_bound_value(4) == 1.0


def test_quantum_strategy_wins_every_question_pair():
    for r in range(3):
        for c in range(3):
            assert np.isclose(cx.quantum_win_probability(r, c), 1.0)
    assert np.isclose(cx.quantum_value(), 1.0)


def test_state_independent_inequality_classical_bound_is_four():
    assert cx.ks_bell_classical_bound() == 4


def test_state_independent_inequality_quantum_value_is_six_for_any_state():
    rng = np.random.default_rng(7)
    for _ in range(10):
        psi = normalize(rng.normal(size=4) + 1j * rng.normal(size=4))
        assert np.isclose(cx.ks_bell_quantum_value(psi), 6.0)
    product = np.kron([1, 0], [1, 0]).astype(complex)  # no entanglement at all
    assert np.isclose(cx.ks_bell_quantum_value(product), 6.0)
