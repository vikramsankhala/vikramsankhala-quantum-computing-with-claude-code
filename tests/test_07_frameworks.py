"""Cross-framework agreement. Qiskit is little-endian, so its bitstrings are reversed."""
import numpy as np
import pytest
from qcwork.circuit import Circuit


def ours():
    return Circuit(3).h(0).cx(0, 1).ry(0.7, 2).cx(1, 2).rz(0.3, 0).probabilities()


def test_qiskit_agrees():
    qiskit = pytest.importorskip("qiskit")
    from qiskit.quantum_info import Statevector
    qc = qiskit.QuantumCircuit(3)
    qc.h(0); qc.cx(0, 1); qc.ry(0.7, 2); qc.cx(1, 2); qc.rz(0.3, 0)
    p = Statevector(qc).probabilities()
    rev = [int(format(i, "03b")[::-1], 2) for i in range(8)]
    assert np.allclose(p[rev], ours(), atol=1e-9)


def test_pennylane_agrees():
    qml = pytest.importorskip("pennylane")
    dev = qml.device("default.qubit", wires=3)

    @qml.qnode(dev)
    def f():
        qml.Hadamard(0); qml.CNOT([0, 1]); qml.RY(0.7, 2); qml.CNOT([1, 2]); qml.RZ(0.3, 0)
        return qml.probs(wires=[0, 1, 2])

    assert np.allclose(f(), ours(), atol=1e-9)


def test_cirq_agrees():
    cirq = pytest.importorskip("cirq")
    q = cirq.LineQubit.range(3)
    c = cirq.Circuit(cirq.H(q[0]), cirq.CNOT(q[0], q[1]), cirq.ry(0.7)(q[2]), cirq.CNOT(q[1], q[2]), cirq.rz(0.3)(q[0]))
    p = np.abs(c.final_state_vector()) ** 2
    assert np.allclose(p, ours(), atol=1e-9)
