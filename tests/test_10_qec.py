import numpy as np
import pytest
from qcwork import qec


def test_repetition_code_exact_values():
    assert np.isclose(qec.repetition_logical_error_exact(0.1, 3), 3 * 0.1 ** 2 * 0.9 + 0.1 ** 3)


def test_repetition_monte_carlo_matches_exact():
    for d in (3, 5, 7):
        mc = qec.repetition_logical_error_mc(0.1, d, shots=400_000, seed=d)
        ex = qec.repetition_logical_error_exact(0.1, d)
        assert abs(mc - ex) < 4 * np.sqrt(ex / 400_000) + 1e-4


def test_larger_distance_helps_only_below_one_half():
    assert qec.repetition_logical_error_exact(0.1, 7) < qec.repetition_logical_error_exact(0.1, 3)
    assert qec.repetition_logical_error_exact(0.6, 7) > qec.repetition_logical_error_exact(0.6, 3)


def test_symplectic_commutation():
    assert qec.symplectic_commute("XX", "ZZ")
    assert not qec.symplectic_commute("XI", "ZI")
    assert qec.symplectic_commute("XZ", "ZX")


def test_steane_code_is_7_1():
    stabs = qec.steane_stabilizers()
    assert len(stabs) == 6
    assert qec.code_parameters(stabs) == (7, 1)


def test_noncommuting_stabilizers_are_rejected():
    with pytest.raises(ValueError):
        qec.code_parameters(["XI", "ZI"])


def test_surface_code_improves_with_distance_below_threshold():
    p = 0.002
    d3 = qec.surface_code_logical_error(3, p, shots=30_000, seed=1)
    d5 = qec.surface_code_logical_error(5, p, shots=30_000, seed=2)
    assert d5 < d3


def test_surface_code_gets_worse_with_distance_above_threshold():
    p = 0.03
    d3 = qec.surface_code_logical_error(3, p, shots=10_000, seed=3)
    d5 = qec.surface_code_logical_error(5, p, shots=10_000, seed=4)
    assert d5 > d3


def test_resource_model_orders_of_magnitude():
    d = qec.distance_for_target(1e-3, 0.01, logical_qubits=100, rounds=1e6)
    assert d % 2 == 1 and d >= 15
    assert qec.physical_qubits(100, d) > 100 * (2 * 15 ** 2 - 1) - 1


def test_resource_model_rejects_near_threshold():
    with pytest.raises(ValueError):
        qec.distance_for_target(0.0099, 1e-9, 100, 1e9)
