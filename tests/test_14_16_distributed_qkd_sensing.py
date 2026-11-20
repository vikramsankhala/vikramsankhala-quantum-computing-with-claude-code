import numpy as np
from qcwork import distributed as d, qkd, sensing, hybrid, tools
from qcwork.operators import I2, X, Y, Z
from qcwork.gates import random_unitary


def rand_state(n, seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
    return v / np.linalg.norm(v)


def test_teleportation_all_four_outcomes():
    psi = rand_state(1, 3)
    for m0 in (0, 1):
        for m1 in (0, 1):
            out, p = d.teleport(psi, m0, m1)
            assert abs(abs(np.vdot(psi, out)) - 1) < 1e-9
            assert abs(p - 0.25) < 1e-9


def test_remote_cnot_all_branches_match_local_cnot():
    psi = rand_state(2, 5)
    target = d.local_cnot(psi)
    for m1 in (0, 1):
        for m2 in (0, 1):
            out, p = d.remote_cnot(psi, m1, m2)
            assert abs(abs(np.vdot(target, out)) - 1) < 1e-9
            assert abs(p - 0.25) < 1e-9


def test_wire_cut_identity():
    rho = np.outer(rand_state(1, 7), rand_state(1, 7).conj())
    assert np.allclose(d.cut_reconstruction(rho), rho)


def test_overheads_and_partition():
    assert d.shot_overhead(wire_cuts=1) == 16 and d.shot_overhead(cnot_cuts=2) == 81
    cut, part = d.min_cut_bipartition(4, [(0, 1), (2, 3), (1, 2)])
    assert cut == 1
    assert d.modules_needed(1000, 300) == 4


def test_bb84_clean_and_eavesdropped():
    clean = qkd.bb84(20000, eve=False, seed=1)
    assert clean["qber"] == 0
    eve = qkd.bb84(20000, eve=True, seed=1)
    assert abs(eve["qber"] - 0.25) < 0.02 and eve["key_fraction"] == 0


def test_qber_threshold_about_11_percent():
    assert abs(qkd.qber_threshold() - 0.110) < 0.002


def test_grover_security():
    assert qkd.grover_effective_security_bits(256) == 128


def test_ghz_heisenberg_scaling():
    for n in (2, 3, 4):
        ghz = sensing.ghz_parity_signal
        assert abs(ghz(n, 0.3) - np.cos(0.3 * n)) < 1e-9
    from qcwork.circuit import ghz_circuit
    q = [sensing.quantum_fisher_information(ghz_circuit(n).state(), n) for n in (2, 3, 4)]
    p = [sensing.quantum_fisher_information(sensing.product_state(n), n) for n in (2, 3, 4)]
    assert np.allclose(q, [4, 9, 16]) and np.allclose(p, [2, 3, 4])


def test_hybrid_helpers():
    assert hybrid.shots_for_precision(1.0, 0.01) == 10000
    assert abs(hybrid.amdahl_speedup(0.5, 1e9) - 2.0) < 1e-6
    loop = hybrid.HybridLoop(lambda p: float((p[0] - 1) ** 2))
    x, f = loop.run([0.0])
    assert abs(x[0] - 1) < 1e-2 and loop.calls > 5


def test_tools_json_interface():
    r = tools.run_circuit(2, [{"gate": "h", "qubits": [0]}, {"gate": "cx", "qubits": [0, 1]}], shots=100)
    assert set(r["probabilities"]) == {"00", "11"} and "OPENQASM" in r["qasm"]
    m = tools.magic_square()
    assert abs(m["classical_value"] - 8 / 9) < 1e-9 and abs(m["quantum_value"] - 1) < 1e-9
    assert tools.estimate_resources(100, 1e-3)["code_distance"] % 2 == 1
