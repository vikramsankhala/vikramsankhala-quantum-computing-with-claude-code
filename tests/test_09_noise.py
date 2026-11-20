import numpy as np
import pytest
from qcwork import noise
from qcwork.qubit import density_matrix, PLUS


def test_channels_are_trace_preserving():
    for K in (noise.depolarizing(0.2), noise.bit_flip(0.1), noise.amplitude_damping(0.3), noise.phase_damping(0.4)):
        assert np.allclose(sum(k.conj().T @ k for k in K), np.eye(2))


def test_amplitude_damping_relaxes_excited_state():
    rho = np.array([[0, 0], [0, 1]], dtype=complex)
    out = noise.apply_kraus(rho, noise.amplitude_damping(0.25), 0, 1)
    assert np.isclose(out[1, 1].real, 0.75)


def test_coherence_decays_as_exp_minus_t_over_T2():
    T1, T2 = 100.0, 60.0
    rho = density_matrix(PLUS)
    out = noise.idle(rho, t=T2, T1=T1, T2=T2, qubit=0, n=1)
    assert np.isclose(abs(out[0, 1]) / abs(rho[0, 1]), np.exp(-1), atol=1e-9)


def test_t2_cannot_exceed_twice_t1():
    with pytest.raises(ValueError):
        noise.t1_t2_channel_params(1.0, T1=10, T2=25)


def test_bell_fidelity_decreases_with_noise():
    f = [noise.noisy_bell_fidelity(p) for p in (0.0, 0.01, 0.05, 0.1)]
    assert np.isclose(f[0], 1.0)
    assert all(a > b for a, b in zip(f, f[1:]))


def test_zero_noise_extrapolation_beats_the_raw_value():
    E0, lam = 1.0, 0.15
    scales = np.array([1.0, 2.0, 3.0])
    values = E0 * np.exp(-lam * scales)
    raw = values[0]
    est = noise.zne_extrapolate(scales, values, order=2)
    assert abs(est - E0) < abs(raw - E0) / 10


def test_readout_mitigation_recovers_true_distribution():
    C = noise.confusion_matrix(0.05, 0.08)
    true = np.array([0.7, 0.3])
    assert np.allclose(noise.readout_mitigate(C @ true, C), true)


def test_randomized_benchmarking_fit_recovers_decay():
    m = np.array([1, 5, 10, 20, 50, 100, 200])
    surv = noise.rb_model(m, 0.5, 0.995, 0.5)
    p, err = noise.rb_fit(m, surv)
    assert np.isclose(p, 0.995, atol=1e-4)
    assert np.isclose(err, 0.0025, atol=1e-4)
