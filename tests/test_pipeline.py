"""Tests for the preprocessing + training pipeline."""

import numpy as np
import pandas as pd

from src.preprocessing import (N_FEATURES, build_dataset, encode_window,
                                 stratified_split)
from src.synthetic_data import FEATURE_VARS, generate_cohort
from src.train import evaluate, train_model


def test_encoding_shape_and_no_nans():
    df = generate_cohort(n_patients=30, seed=3)
    X, y, ids = build_dataset(df)
    assert X.shape[1] == N_FEATURES == len(FEATURE_VARS) * 8 + 1
    assert not X.isna().any().any(), "encoded features must have no NaNs"
    assert set(y.unique()) <= {0, 1}
    assert len(y) == len(ids) == X.shape[0]


def test_encode_all_missing_variable():
    df = generate_cohort(n_patients=10, seed=5)
    grp = df[df["patient_id"] == 0].copy()
    grp["wbc"] = np.nan  # entirely missing variable
    feats = encode_window(grp, end_hour=48)
    assert feats["wbc__count"] == 0.0
    assert feats["wbc__mean"] == 0.0


def test_stratified_split_ratios():
    df = generate_cohort(n_patients=200, seed=9)
    X, y, ids = build_dataset(df)
    splits = stratified_split(X, y, ids, seed=9)
    total = sum(len(v[1]) for v in splits.values())
    assert total == len(y)
    for name, frac in (("train", 0.7), ("val", 0.15), ("test", 0.15)):
        assert abs(len(splits[name][1]) / len(y) - frac) < 0.05, name
    # Case rate roughly preserved across splits.
    rates = [splits[n][1].mean() for n in ("train", "val", "test")]
    assert max(rates) - min(rates) < 0.15


def test_end_to_end_small():
    """Full run: generate -> encode -> train -> evaluate."""
    df = generate_cohort(n_patients=250, seed=13)
    X, y, ids = build_dataset(df, seed=13)
    splits = stratified_split(X, y, ids, seed=13)
    X_train, y_train, _ = splits["train"]
    X_test, y_test, _ = splits["test"]
    model = train_model(X_train, y_train, seed=13, n_estimators=100)
    metrics = evaluate(model, X_test, y_test)
    assert metrics["auroc"] > 0.7, metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
