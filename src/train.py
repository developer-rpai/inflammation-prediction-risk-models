"""End-to-end training for inflammation prediction on synthetic ICU data.

Usage:
    python src/train.py                      # synthetic data, default settings
    python src/train.py --n-patients 1000 --seed 7
    python src/train.py --model gru          # recurrent (GRU) path
    python src/train.py --data-dir input/xgboost   # pre-extracted real data

With --data-dir, the script expects a long-format parquet/csv with columns
patient_id, hour, the 44 clinical variables (see src/synthetic_data.py),
and onset_hour -- i.e. the same schema generate_cohort() produces. Without
it, a synthetic cohort is generated on the fly (no credentials needed).

Two model families are supported:

* ``xgboost`` (default): gradient-boosted trees on per-variable summary
  statistics (count, mean, std, min, max, quartiles) over the 48-hour
  observation window -- the same encoding documented in the README. Falls
  back to scikit-learn's GradientBoostingClassifier when xgboost is not
  installed.
* ``gru``: a single-layer GRU (NumPy implementation in src/gru.py, no
  deep-learning framework required) trained directly on the raw 48-hour
  hourly sequences (forward-filled, median-imputed, z-scored with
  training-set statistics).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import build_dataset, stratified_split
from src.sequences import (build_sequences, impute_and_standardize,
                           sequence_split)
from src.synthetic_data import generate_cohort

try:
    from xgboost import XGBClassifier

    _XGB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _XGB_AVAILABLE = False

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score


def load_cohort(data_dir: str | None, n_patients: int, seed: int) -> pd.DataFrame:
    if data_dir:
        for fname in ("cohort.parquet", "cohort.csv"):
            path = os.path.join(data_dir, fname)
            if os.path.exists(path):
                print(f"Loading pre-extracted cohort from {path}")
                return (pd.read_parquet(path) if fname.endswith("parquet")
                        else pd.read_csv(path))
        raise FileNotFoundError(
            f"--data-dir {data_dir} has no cohort.parquet/cohort.csv")
    print(f"Generating synthetic cohort: {n_patients} patients, seed={seed}")
    return generate_cohort(n_patients=n_patients, seed=seed)


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, seed: int,
                  n_estimators: int = 300):
    if _XGB_AVAILABLE:
        model = XGBClassifier(
            n_estimators=n_estimators, max_depth=4, learning_rate=0.05,
            subsample=0.7, colsample_bytree=0.7, reg_lambda=5.0,
            min_child_weight=3, n_jobs=-1, random_state=seed,
            eval_metric="logloss",
        )
    else:
        warnings.warn("xgboost not installed; using sklearn "
                      "GradientBoostingClassifier instead.")
        model = GradientBoostingClassifier(random_state=seed)
    model.fit(X_train, y_train)
    return model


def train_gru(X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray, seed: int,
              n_hidden: int = 32, epochs: int = 15, lr: float = 3e-3,
              batch_size: int = 64):
    from src.gru import GRUClassifier
    model = GRUClassifier(n_features=X_train.shape[2], n_hidden=n_hidden,
                          seed=seed, lr=lr, epochs=epochs,
                          batch_size=batch_size, verbose=True)
    model.fit(X_train, y_train, X_val, y_val)
    return model


def evaluate(model, X, y) -> dict[str, float]:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "auroc": float(roc_auc_score(y, proba)),
        "auprc": float(average_precision_score(y, proba)),
        "n": int(len(y)),
        "prevalence": float(np.asarray(y).mean()),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-patients", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", default=None,
                    help="Directory with cohort.parquet/csv; else synthetic.")
    ap.add_argument("--out-dir", default="output/synthetic")
    ap.add_argument("--n-estimators", type=int, default=300,
                    help="Trees for the xgboost path.")
    ap.add_argument("--model", choices=("xgboost", "gru"), default="xgboost",
                    help="Model family: gradient boosting or recurrent GRU.")
    ap.add_argument("--gru-hidden", type=int, default=32)
    ap.add_argument("--gru-epochs", type=int, default=15)
    ap.add_argument("--gru-lr", type=float, default=3e-3)
    ap.add_argument("--gru-batch", type=int, default=64)
    args = ap.parse_args()

    cohort = load_cohort(args.data_dir, args.n_patients, args.seed)

    if args.model == "gru":
        out_subdir = os.path.join(args.out_dir, "gru")
        X, y, ids = build_sequences(cohort, seed=args.seed)
        print(f"dataset: {X.shape[0]} patients, {X.shape[1]}h x "
              f"{X.shape[2]} vars sequences, case rate {y.mean():.3f}")
        splits = sequence_split(X, y, ids, seed=args.seed)
        (X_train, y_train, _), (X_val, y_val, _), (X_test, y_test, _) = (
            splits["train"], splits["val"], splits["test"])
        X_train, X_val, X_test = impute_and_standardize(X_train, X_val, X_test)
        print(f"split sizes: train={len(y_train)} val={len(y_val)} "
              f"test={len(y_test)}")
        model = train_gru(X_train, y_train, X_val, y_val, args.seed,
                          n_hidden=args.gru_hidden, epochs=args.gru_epochs,
                          lr=args.gru_lr, batch_size=args.gru_batch)
        model_name = (f"GRUClassifier(hidden={args.gru_hidden})")
        importances = None
    else:
        out_subdir = os.path.join(args.out_dir, "xgboost")
        X, y, ids = build_dataset(cohort)
        print(f"dataset: {X.shape[0]} patients, {X.shape[1]} features, "
              f"case rate {y.mean():.3f}")
        splits = stratified_split(X, y, ids, seed=args.seed)
        X_train, y_train, _ = splits["train"]
        X_val, y_val, _ = splits["val"]
        X_test, y_test, _ = splits["test"]
        print(f"split sizes: train={len(y_train)} val={len(y_val)} "
              f"test={len(y_test)}")
        model = train_xgboost(X_train, y_train, args.seed, args.n_estimators)
        model_name = type(model).__name__
        importances = pd.DataFrame({
            "feature": X.columns,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)

    metrics = {
        "train": evaluate(model, X_train, y_train),
        "val": evaluate(model, X_val, y_val),
        "test": evaluate(model, X_test, y_test),
        "config": {"n_patients": args.n_patients, "seed": args.seed,
                   "synthetic": args.data_dir is None,
                   "model": model_name},
    }
    for split in ("train", "val", "test"):
        m = metrics[split]
        print(f"{split:5s}  acc={m['accuracy']:.3f}  "
              f"auroc={m['auroc']:.3f}  auprc={m['auprc']:.3f}  "
              f"(n={m['n']}, prev={m['prevalence']:.3f})")

    os.makedirs(out_subdir, exist_ok=True)
    with open(os.path.join(out_subdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    if importances is not None:
        importances.to_csv(os.path.join(out_subdir, "feature_importances.csv"),
                           index=False)
        if _XGB_AVAILABLE and hasattr(model, "save_model"):
            model.save_model(os.path.join(out_subdir, "model.json"))
        print("\nTop 10 features:")
        print(importances.head(10).to_string(index=False))
    print(f"\nArtifacts written to {out_subdir}/")


if __name__ == "__main__":
    main()
