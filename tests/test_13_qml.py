import numpy as np
from qcwork import qml


def test_kernel_is_psd_with_unit_diagonal():
    X, _ = qml.make_blobs(6)
    K = qml.quantum_kernel(X)
    assert np.allclose(np.diag(K), 1) and np.allclose(K, K.T)
    assert np.linalg.eigvalsh(K).min() > -1e-9


def test_rbf_classifier_separates_blobs():
    X, y = qml.make_blobs(15, 0)
    Xt, yt = qml.make_blobs(30, 1)
    m = qml.KernelRidge(lambda A, B=None: qml.rbf_kernel(A, B)).fit(X, y)
    assert (m.predict(Xt) == yt).mean() > 0.95


def test_quantum_kernel_beats_chance():
    X, y = qml.make_blobs(15, 0)
    Xt, yt = qml.make_blobs(30, 1)
    m = qml.KernelRidge(lambda A, B=None: qml.quantum_kernel(A, B)).fit(X, y)
    assert (m.predict(Xt) == yt).mean() > 0.7


def test_reupload_learns():
    X, y = qml.make_blobs(15, 0)
    Xt, yt = qml.make_blobs(30, 1)
    p, loss = qml.train_reupload(X, y, 3, 150, 0.3, 0)
    assert loss < 0.1 and (qml.reupload_predict(Xt, p) == yt).mean() > 0.9
