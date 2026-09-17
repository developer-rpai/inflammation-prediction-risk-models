"""End-to-end training for inflammation prediction on synthetic ICU data.

Usage:
    python src/train.py                      # synthetic data, default settings
    python src/train.py --n-patients 1000 --seed 7
    python src/train.py --data-dir input/xgboost   # pre-extracted real data

With --data-dir, the script expects a long-format parquet/csv with columns
patient_id, hour, the 44 clinical variables (see src/synthetic_data.py),
and onset_hour -- i.e. the same schema generate_cohort() produces. Without
it, a synthetic cohort is generated on the fly (no credentials needed).

The model is gradient-boosted trees (XGBoost), matching the approach
documented in the README. If xgboost is not installed, it falls back to
scikit-learn's GradientBoostingClassifier with a warning.
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


def train_model(X_train: pd.DataFrame, y_train: pd.Series, seed: int,
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


def evaluate(model, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "auroc": float(roc_auc_score(y, proba)),
        "auprc": float(average_precision_score(y, proba)),
        "n": int(len(y)),
        "prevalence": float(y.mean()),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-patients", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", default=None,
                    help="Directory with cohort.parquet/csv; else synthetic.")
    ap.add_argument("--out-dir", default="output/synthetic")
    ap.add_argument("--n-estimators", type=int, default=300)
    args = ap.parse_args()

    cohort = load_cohort(args.data_dir, args.n_patients, args.seed)
    X, y, ids = build_dataset(cohort)
    print(f"dataset: {X.shape[0]} patients, {X.shape[1]} features, "
          f"case rate {y.mean():.3f}")
    splits = stratified_split(X, y, ids, seed=args.seed)
    X_train, y_train, _ = splits["train"]
    X_val, y_val, _ = splits["val"]
    X_test, y_test, _ = splits["test"]
    print(f"split sizes: train={len(y_train)} val={len(y_val)} "
          f"test={len(y_test)}")

    model = train_model(X_train, y_train, args.seed, args.n_estimators)

    metrics = {
        "train": evaluate(model, X_train, y_train),
        "val": evaluate(model, X_val, y_val),
        "test": evaluate(model, X_test, y_test),
        "config": {"n_patients": args.n_patients, "seed": args.seed,
                   "synthetic": args.data_dir is None,
                   "model": type(model).__name__},
    }
    for split in ("train", "val", "test"):
        m = metrics[split]
        print(f"{split:5s}  acc={m['accuracy']:.3f}  "
              f"auroc={m['auroc']:.3f}  auprc={m['auprc']:.3f}  "
              f"(n={m['n']}, prev={m['prevalence']:.3f})")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    importances = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importances.to_csv(os.path.join(args.out_dir, "feature_importances.csv"),
                       index=False)
    if _XGB_AVAILABLE:
        model.save_model(os.path.join(args.out_dir, "model.json"))
    print("\nTop 10 features:")
    print(importances.head(10).to_string(index=False))
    print(f"\nArtifacts written to {args.out_dir}/")


if __name__ == "__main__":
    main()
