import numpy as np
from qcwork import entanglement as E
from qcwork.qubit import bell_state


def test_chsh_tsirelson_bound():
    assert abs(E.chsh_value() - 2 * np.sqrt(2)) < 1e-12
    assert abs(E.chsh_from_state(bell_state(0)) - 2 * np.sqrt(2)) < 1e-9


def test_chsh_visibility_threshold_is_1_over_sqrt2():
    assert E.chsh_value(1 / np.sqrt(2) - 1e-6) < 2 < E.chsh_value(1 / np.sqrt(2) + 1e-6)


def test_product_state_obeys_classical_bound():
    zero = np.array([1, 0], dtype=complex)
    assert E.chsh_from_state(np.kron(zero, zero)) <= 2 + 1e-9


def test_e91_clean_violates_chsh():
    r = E.e91(60000, eve=False, seed=3)
    assert r["S"] > 2.7 and r["qber"] < 0.01 and r["secure"]


def test_e91_eavesdropper_breaks_chsh_and_adds_errors():
    r = E.e91(60000, eve=True, seed=3)
    assert r["S"] < 2.0 and not r["secure"] and r["qber"] > 0.1


def test_superdense_all_four_messages():
    for b1 in (0, 1):
        for b0 in (0, 1):
            assert E.superdense_encode_decode(b1, b0) == (b1, b0)


def test_entanglement_swapping_all_branches():
    for m1 in (0, 1):
        for m2 in (0, 1):
            psi, p = E.entanglement_swap(m1, m2)
            assert abs(abs(np.vdot(bell_state(0), psi)) - 1) < 1e-9
            assert abs(p - 0.25) < 1e-9


def test_ghz_correlators_and_sharing():
    c = E.ghz_correlators()
    for k, s in E.GHZ_SIGNS.items():
        assert abs(c[k] - s) < 1e-9
    r = E.ghz_secret_sharing(1500, seed=1)
    assert r["kept"] > 200 and r["consistent"] == r["kept"]
