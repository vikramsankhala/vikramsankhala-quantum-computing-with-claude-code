import numpy as np
import pytest
from qcwork import gates as g
from qcwork.circuit import Circuit, bell_circuit, ghz_circuit
from qcwork.operators import is_unitary, X, Z


@pytest.mark.parametrize("U", [g.H, g.S, g.T, g.CNOT, g.CZ, g.SWAP, g.TOFFOLI, g.rx(0.3), g.ry(1.2), g.rz(2.0)])
def test_every_gate_is_unitary(U):
    assert is_unitary(U)


def test_standard_identities():
    assert np.allclose(g.H @ g.H, np.eye(2))
    assert np.allclose(g.H @ X @ g.H, Z)
    assert np.allclose(g.S @ g.S, Z)
    assert np.allclose(g.T @ g.T, g.S)
    assert np.allclose(g.SWAP, g.CNOT @ np.kron(g.H, g.H) @ g.CNOT @ np.kron(g.H, g.H) @ g.CNOT)


def test_zyz_decomposition_reconstructs_any_single_qubit_gate():
    rng = np.random.default_rng(4)
    gates = [g.H, g.S, g.T, g.X if hasattr(g, "X") else X] + [g.random_unitary(2, rng) for _ in range(30)]
    for U in gates:
        assert np.allclose(g.from_zyz(*g.zyz_decompose(U)), U, atol=1e-9)


def test_bell_circuit_produces_bell_state():
    assert np.allclose(bell_circuit().state(), [2 ** -0.5, 0, 0, 2 ** -0.5])


def test_ghz_circuit():
    psi = ghz_circuit(4).state()
    assert np.isclose(psi[0], 2 ** -0.5) and np.isclose(psi[-1], 2 ** -0.5)


def test_circuit_unitary_matches_matrix_product():
    c = Circuit(2).h(0).cx(0, 1)
    assert np.allclose(c.unitary(), g.CNOT @ np.kron(g.H, np.eye(2)))


def test_gate_on_nonadjacent_qubits_matches_embedding():
    c = Circuit(3).cx(0, 2)
    expected = np.zeros((8, 8))
    for i in range(8):
        a, b, t = (i >> 2) & 1, (i >> 1) & 1, i & 1
        expected[(a << 2) | (b << 1) | (t ^ a), i] = 1
    assert np.allclose(c.unitary(), expected)


def test_measurement_collapse_and_forced_outcome():
    c = bell_circuit()
    assert c.measure(0, outcome=1) == 1
    assert np.allclose(c.probabilities(), [0, 0, 0, 1])
    with pytest.raises(ValueError):
        bell_circuit().measure(0, outcome=1) and Circuit(1).measure(0, outcome=1)


def test_sampling_statistics():
    counts = bell_circuit().sample(10000, seed=1)
    assert set(counts) == {"00", "11"}
    assert abs(counts["00"] / 10000 - 0.5) < 0.03


def test_depth_and_gate_count():
    c = Circuit(3).h(0).h(1).cx(0, 1).cx(1, 2)
    assert c.depth() == 3
    assert c.gate_count() == {"h": 2, "cx": 2}


def test_expectation_of_pauli_strings():
    c = bell_circuit()
    assert np.isclose(c.expectation("ZZ"), 1.0)
    assert np.isclose(c.expectation("XX"), 1.0)
    assert np.isclose(c.expectation("ZI"), 0.0)


def test_qasm_export():
    qasm = bell_circuit().to_qasm()
    assert "h q[0];" in qasm and "cx q[0],q[1];" in qasm


def test_clifford_check():
    assert g.clifford_check(g.H) and g.clifford_check(g.CNOT)
    assert not g.clifford_check(g.T)
