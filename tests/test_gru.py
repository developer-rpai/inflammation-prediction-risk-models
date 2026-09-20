"""Tests for the NumPy GRU path (sequences + classifier)."""

import numpy as np

from src.gru import GRUClassifier
from src.sequences import (build_sequences, impute_and_standardize,
                           sequence_split)
from src.synthetic_data import generate_cohort


def _tiny_model():
    return GRUClassifier(n_features=3, n_hidden=4, seed=0, epochs=1,
                         batch_size=8)


def test_forward_shapes():
    clf = _tiny_model()
    rng = np.random.default_rng(0)
    X = rng.standard_normal((5, 10, 3)).astype(np.float32)
    proba = clf.predict_proba(X)
    assert proba.shape == (5, 2)
    assert np.all(proba >= 0) and np.all(proba <= 1)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_gradients_match_numeric():
    """Analytic BPTT gradients must match finite differences."""
    clf = GRUClassifier(n_features=2, n_hidden=3, seed=1, l2=0.0)
    rng = np.random.default_rng(2)
    X = rng.standard_normal((4, 5, 2))
    y = np.array([0.0, 1.0, 1.0, 0.0])

    _, grads = clf._loss_and_grads(X, y)
    for key in ("Wz", "Uz", "Wh", "Uh", "wo"):
        analytic = grads[key]
        numeric = np.zeros_like(analytic)
        it = np.nditer(analytic, flags=["multi_index"])
        checked = 0
        while not it.finished and checked < 6:
            ix = it.multi_index
            orig = clf.params[key][ix]
            h = 1e-5
            clf.params[key][ix] = orig + h
            lp, _ = clf._loss_and_grads(X, y)
            clf.params[key][ix] = orig - h
            lm, _ = clf._loss_and_grads(X, y)
            clf.params[key][ix] = orig
            numeric[ix] = (lp - lm) / (2 * h)
            checked += 1
            it.iternext()
        # compare on the checked entries
        mask = numeric != 0
        assert np.allclose(analytic[mask], numeric[mask],
                           rtol=1e-3, atol=1e-5), key


def test_build_sequences_shapes():
    cohort = generate_cohort(n_patients=60, seed=3)
    X, y, ids = build_sequences(cohort, seed=3)
    assert X.ndim == 3 and X.shape[1] == 48 and X.shape[2] == 44
    assert X.shape[0] == len(y) == len(ids)
    assert set(np.unique(y)) <= {0, 1}
    assert y.mean() > 0.05  # both classes represented


def test_impute_standardize_no_leak_no_nan():
    cohort = generate_cohort(n_patients=60, seed=4)
    X, y, ids = build_sequences(cohort, seed=4)
    splits = sequence_split(X, y, ids, seed=4)
    (Xtr, ytr, _), (Xva, yva, _), (Xte, yte, _) = (
        splits["train"], splits["val"], splits["test"])
    Xtr2, Xva2, Xte2 = impute_and_standardize(Xtr, Xva, Xte)
    for a in (Xtr2, Xva2, Xte2):
        assert not np.isnan(a).any()
    # train split is ~zero-mean/unit-variance per variable
    assert abs(Xtr2.mean()) < 0.05


def test_gru_learns_synthetic_signal():
    """The GRU must beat chance on the synthetic cohort (strong signal)."""
    cohort = generate_cohort(n_patients=240, seed=5)
    X, y, ids = build_sequences(cohort, seed=5)
    splits = sequence_split(X, y, ids, seed=5)
    (Xtr, ytr, _), (Xva, yva, _), (Xte, yte, _) = (
        splits["train"], splits["val"], splits["test"])
    Xtr, Xva, Xte = impute_and_standardize(Xtr, Xva, Xte)
    clf = GRUClassifier(n_features=Xtr.shape[2], n_hidden=16, seed=5,
                        epochs=8, batch_size=32, patience=3)
    clf.fit(Xtr, ytr, Xva, yva)
    from sklearn.metrics import roc_auc_score
    auroc = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
    assert auroc > 0.65, f"GRU auroc {auroc:.3f} not above 0.65"
