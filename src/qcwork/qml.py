"""Module 13: quantum machine learning: kernels and a data re-uploading classifier."""
from __future__ import annotations

import numpy as np

from .circuit import Circuit


def feature_map_state(x, reps: int = 2) -> np.ndarray:
    """ZZ-style feature map: H, then phase rotations by x and pairwise products."""
    n = len(x)
    c = Circuit(n)
    for _ in range(reps):
        for q in range(n):
            c.h(q)
            c.rz(2 * x[q], q)
        for q in range(n - 1):
            c.cx(q, q + 1)
            c.rz(2 * (np.pi - x[q]) * (np.pi - x[q + 1]), q + 1)
            c.cx(q, q + 1)
    return c.state()


def quantum_kernel(X1, X2=None, reps: int = 2) -> np.ndarray:
    """K_ij = |<phi(x_i)|phi(y_j)>|^2."""
    X2 = X1 if X2 is None else X2
    A = np.array([feature_map_state(x, reps) for x in X1])
    B = np.array([feature_map_state(x, reps) for x in X2])
    return np.abs(A.conj() @ B.T) ** 2


def rbf_kernel(X1, X2=None, gamma: float = 1.0) -> np.ndarray:
    X2 = X1 if X2 is None else X2
    d = ((np.asarray(X1)[:, None, :] - np.asarray(X2)[None, :, :]) ** 2).sum(-1)
    return np.exp(-gamma * d)


class KernelRidge:
    """Kernel ridge classifier: works with any kernel function."""

    def __init__(self, kernel, lam: float = 1e-3):
        self.kernel, self.lam = kernel, lam

    def fit(self, X, y):
        self.X = np.asarray(X)
        K = self.kernel(self.X)
        self.alpha = np.linalg.solve(K + self.lam * np.eye(len(K)), np.asarray(y, dtype=float))
        return self

    def predict(self, X):
        return np.sign(self.kernel(np.asarray(X), self.X) @ self.alpha)


def make_blobs(n_per_class: int, seed: int = 0):
    """Two clusters in [0, pi]^2 with labels -1 and +1."""
    rng = np.random.default_rng(seed)
    a = rng.normal([0.9, 0.9], 0.25, size=(n_per_class, 2))
    b = rng.normal([2.2, 2.2], 0.25, size=(n_per_class, 2))
    X = np.clip(np.vstack([a, b]), 0, np.pi)
    y = np.array([-1] * n_per_class + [1] * n_per_class)
    return X, y


# --------------------------- single-qubit data re-uploading classifier
def reupload_output(x, params) -> float:
    """<Z> after L layers of RZ(w0 x0 + b0) RY(w1 x1 + b1) on one qubit."""
    L = len(params) // 4
    c = Circuit(1)
    for l in range(L):
        w0, b0, w1, b1 = params[4 * l: 4 * l + 4]
        c.rz(w0 * x[0] + b0, 0)
        c.ry(w1 * x[1] + b1, 0)
    return c.expectation("Z")


def train_reupload(X, y, layers: int = 3, steps: int = 200, lr: float = 0.2, seed: int = 0):
    rng = np.random.default_rng(seed)
    p = rng.normal(0, 1, 4 * layers)
    loss = lambda q: np.mean([(reupload_output(x, q) - t) ** 2 for x, t in zip(X, y)])
    for _ in range(steps):
        g = np.zeros_like(p)
        for k in range(len(p)):
            e = np.zeros_like(p)
            e[k] = 1e-4
            g[k] = (loss(p + e) - loss(p - e)) / 2e-4
        p = p - lr * g
    return p, loss(p)


def reupload_predict(X, params):
    return np.sign([reupload_output(x, params) for x in X])
