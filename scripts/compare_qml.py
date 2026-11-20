"""Compare three classifiers on the same data across seeds (Module 13, Lab 13).

Usage: python scripts/compare_qml.py [n_seeds]
Reports mean, standard deviation, min and max test accuracy. Neither quantum model
is tuned harder than the classical one.
"""
import sys
import numpy as np
from qcwork import qml


def main(n_seeds: int = 10):
    res = {"quantum kernel": [], "RBF kernel": [], "re-uploading": []}
    for s in range(n_seeds):
        X, y = qml.make_blobs(15, seed=100 + s)
        Xt, yt = qml.make_blobs(30, seed=200 + s)
        qk = qml.KernelRidge(lambda A, B=None: qml.quantum_kernel(A, B)).fit(X, y)
        rb = qml.KernelRidge(lambda A, B=None: qml.rbf_kernel(A, B)).fit(X, y)
        p, _ = qml.train_reupload(X, y, 3, 150, 0.3, seed=s)
        res["quantum kernel"].append((qk.predict(Xt) == yt).mean())
        res["RBF kernel"].append((rb.predict(Xt) == yt).mean())
        res["re-uploading"].append((qml.reupload_predict(Xt, p) == yt).mean())
    for k, v in res.items():
        print(f"{k:15s} mean {np.mean(v):.3f}  sd {np.std(v):.3f}  min {min(v):.3f}  max {max(v):.3f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
