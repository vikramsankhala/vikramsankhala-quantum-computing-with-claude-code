"""Applications of entanglement: CHSH, E91 key distribution, superdense coding,
entanglement swapping and GHZ secret sharing (Modules 3 and 16)."""
from __future__ import annotations

import numpy as np

from .circuit import Circuit, ghz_circuit
from .distributed import _extract
from .qubit import bell_state

# Observable along angle a in the x-z plane: A(a) = cos(a) Z + sin(a) X.
# For the Bell state |Phi+>, <A(a) (x) B(b)> = cos(a - b).


def correlator(a: float, b: float, noise_visibility: float = 1.0) -> float:
    """E(a, b) for |Phi+> mixed with white noise of the given visibility."""
    return noise_visibility * np.cos(a - b)


def chsh_value(visibility: float = 1.0) -> float:
    """CHSH S = E(a0,b0) + E(a0,b1) + E(a1,b0) - E(a1,b1) at the optimal angles.
    Local hidden-variable models obey |S| <= 2; quantum mechanics reaches 2 sqrt(2)."""
    a0, a1, b0, b1 = 0.0, np.pi / 2, np.pi / 4, -np.pi / 4
    return float(correlator(a0, b0, visibility) + correlator(a0, b1, visibility)
                 + correlator(a1, b0, visibility) - correlator(a1, b1, visibility))


def chsh_from_state(psi: np.ndarray) -> float:
    """CHSH value computed directly from a two-qubit state vector."""
    from .operators import X, Z

    def obs(a):
        return np.cos(a) * Z + np.sin(a) * X

    a0, a1, b0, b1 = 0.0, np.pi / 2, np.pi / 4, -np.pi / 4
    e = lambda a, b: float(np.vdot(psi, np.kron(obs(a), obs(b)) @ psi).real)
    return e(a0, b0) + e(a0, b1) + e(a1, b0) - e(a1, b1)


def _sample_pair(a: float, b: float, rng, mode: str):
    """Sample one +-1 outcome pair. mode: 'ideal' (|Phi+>) or 'eve' (Z-basis intercept-resend)."""
    if mode == "ideal":
        p_equal = (1 + np.cos(a - b)) / 2
    else:  # Eve measured Z and resent |00> or |11>: <A(a)B(b)> = cos(a) cos(b)
        p_equal = (1 + np.cos(a) * np.cos(b)) / 2
    x = 1 if rng.random() < 0.5 else -1
    y = x if rng.random() < p_equal else -x
    return x, y


def e91(rounds: int = 60000, eve: bool = False, seed: int = 0) -> dict:
    """Simulate E91. Alice picks angles from {0, pi/4, pi/2}, Bob from {pi/4, pi/2, 3pi/4}.
    Matching angles (pi/4, pi/2) give key bits; the four CHSH-relevant pairs give S."""
    rng = np.random.default_rng(seed)
    A = [0.0, np.pi / 4, np.pi / 2]
    B = [np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    mode = "eve" if eve else "ideal"
    key_a, key_b = [], []
    sums = {}
    counts = {}
    for _ in range(rounds):
        ia, ib = rng.integers(0, 3), rng.integers(0, 3)
        a, b = A[ia], B[ib]
        x, y = _sample_pair(a, b, rng, mode)
        if np.isclose(a, b):
            key_a.append(x)
            key_b.append(y)
        # CHSH pairs: Alice in {0, pi/2}, Bob in {pi/4, 3pi/4}
        if ia in (0, 2) and ib in (0, 2):
            k = (ia, ib)
            sums[k] = sums.get(k, 0) + x * y
            counts[k] = counts.get(k, 0) + 1
    E = {k: sums[k] / counts[k] for k in sums}
    S = E[(0, 0)] - E[(0, 2)] + E[(2, 0)] + E[(2, 2)]
    ka, kb = np.array(key_a), np.array(key_b)
    # E91 key bits: Bob flips his bit when the pair is anticorrelated by convention; here |Phi+> is correlated.
    qber = float(np.mean(ka != kb)) if len(ka) else float("nan")
    return {"S": float(S), "qber": qber, "key_bits": int(len(ka)), "secure": bool(S > 2)}


# ----------------------------------------------------------- superdense coding
def superdense_encode_decode(b1: int, b0: int) -> tuple[int, int]:
    """Alice sends two classical bits by acting on her half of a Bell pair with one qubit."""
    c = Circuit(2).h(0).cx(0, 1)          # shared Bell pair; Alice holds qubit 0
    if b0:
        c.x(0)
    if b1:
        c.z(0)
    c.cx(0, 1).h(0)                        # Bob's Bell-basis measurement
    probs = c.probabilities()
    k = int(np.argmax(probs))
    assert probs[k] > 1 - 1e-9
    return (k >> 1) & 1, k & 1


# --------------------------------------------------------- entanglement swapping
def entanglement_swap(m1: int, m2: int):
    """Pairs (0,1) and (2,3). A Bell measurement on (1,2) entangles 0 and 3 without
    those qubits ever interacting. Returns (state of qubits 0 and 3 after correction, probability)."""
    c = Circuit(4).h(0).cx(0, 1).h(2).cx(2, 3)
    c.cx(1, 2).h(1)
    p = float(np.sum(np.abs(c.state().reshape([2] * 4)[:, m1, m2, :]) ** 2))
    c.measure(1, outcome=m1)
    c.measure(2, outcome=m2)
    if m2:
        c.x(3)
    if m1:
        c.z(3)
    return _extract(c, {1: m1, 2: m2}), p


# ------------------------------------------------------------ GHZ secret sharing
GHZ_SIGNS = {"XXX": 1, "XYY": -1, "YXY": -1, "YYX": -1}


def ghz_correlators() -> dict:
    return {k: ghz_circuit(3).expectation(k) for k in GHZ_SIGNS}


def ghz_secret_sharing(rounds: int = 2000, seed: int = 0) -> dict:
    """Three parties share a GHZ state and each measure X or Y at random. In rounds whose bases
    are one of XXX, XYY, YXY, YYX the product of outcomes is fixed, so any two parties can
    reconstruct the third party's bit only by combining their results."""
    rng = np.random.default_rng(seed)
    kept = agree = 0
    for _ in range(rounds):
        bases = "".join(rng.choice(["X", "Y"], 3))
        if bases not in GHZ_SIGNS:
            continue
        c = ghz_circuit(3)
        for q, bs in enumerate(bases):
            if bs == "Y":
                c.sdg(q)
            c.h(q)
        out = list(c.sample(1, seed=int(rng.integers(1 << 30))))[0]
        r = [1 - 2 * int(b) for b in out]
        kept += 1
        agree += (r[0] * r[1] * r[2] == GHZ_SIGNS[bases])
    return {"kept": kept, "consistent": agree}
