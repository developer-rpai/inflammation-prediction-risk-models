"""Preprocessing for the synthetic inflammation-prediction pipeline.

Mirrors the encoding scheme documented in the README for the gradient-boosting
models: each variable's time series is summarized into distribution statistics
(count, mean, std, min, max, quartiles), because tree models cannot consume
raw sequences directly. Statistics are NaN-aware; any variable that is
entirely missing for a patient contributes zeros.

Task framing (early prediction, 7 h horizon): for each ICU stay we choose a
prediction time T and use the preceding ``OBS_WINDOW`` hours as model input.
Cases are predicted at T = onset - 7 h, i.e. the model must flag inflammation
7 hours before the computable-phenotype onset -- the same horizon as the
original study. Controls are predicted at a random time T with a full history
and a 48 h onset-free follow-up. Cases whose onset is too early for a full
observation window are excluded, mirroring the original study's exclusion of
very-early onsets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .synthetic_data import FEATURE_VARS

#: Hours of history used as model input.
OBS_WINDOW = 48
#: Prediction horizon: cases are predicted this many hours before onset.
PREDICTION_HORIZON = 7
#: Controls must remain onset-free this long after the prediction time.
FOLLOWUP_WINDOW = 48

_STAT_NAMES = ("count", "mean", "std", "min", "max", "q25", "q50", "q75")


def _encode_series(s: pd.Series) -> dict[str, float]:
    vals = s.to_numpy(dtype=float)
    count = int(np.sum(~np.isnan(vals)))
    if count == 0:
        return {stat: 0.0 for stat in _STAT_NAMES}
    return {
        "count": float(count),
        "mean": float(np.nanmean(vals)),
        "std": float(np.nanstd(vals)) if count > 1 else 0.0,
        "min": float(np.nanmin(vals)),
        "max": float(np.nanmax(vals)),
        "q25": float(np.nanquantile(vals, 0.25)),
        "q50": float(np.nanquantile(vals, 0.50)),
        "q75": float(np.nanquantile(vals, 0.75)),
    }


def encode_window(df_patient: pd.DataFrame, end_hour: int,
                  obs_window: int = OBS_WINDOW) -> dict[str, float]:
    """Encode the ``obs_window`` hours ending at ``end_hour`` into features."""
    window = df_patient[(df_patient["hour"] >= end_hour - obs_window)
                        & (df_patient["hour"] < end_hour)]
    feats: dict[str, float] = {"n_hours": float(len(window))}
    for var in FEATURE_VARS:
        for stat, value in _encode_series(window[var]).items():
            feats[f"{var}__{stat}"] = value
    return feats


def build_dataset(cohort: pd.DataFrame, seed: int = 42,
                  obs_window: int = OBS_WINDOW,
                  horizon: int = PREDICTION_HORIZON,
                  followup: int = FOLLOWUP_WINDOW
                  ) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Build the modeling dataset from a long-format cohort frame.

    Returns (X, y, patient_ids). y is 1 when inflammation onset occurs
    ``horizon`` hours after the prediction time, 0 for controls sampled at
    a random prediction time with an onset-free follow-up. Patients that
    cannot supply a full observation window are excluded.
    """
    rng = np.random.default_rng(seed)
    rows, labels, ids = [], [], []
    for pid, grp in cohort.groupby("patient_id", sort=True):
        grp = grp.sort_values("hour")
        n = len(grp)
        onset = grp["onset_hour"].iloc[0]
        if not np.isnan(onset):
            pred_time = int(onset) - horizon
            if pred_time < obs_window:
                continue  # onset too early for a full observation window
            label = 1
        else:
            lo, hi = obs_window, n - followup
            if hi <= lo:
                continue  # stay too short for history + follow-up
            pred_time = int(rng.integers(lo, hi))
            label = 0
        rows.append(encode_window(grp, pred_time, obs_window))
        labels.append(label)
        ids.append(pid)
    X = pd.DataFrame(rows)
    y = pd.Series(labels, dtype=int, name="label")
    return X, y, np.asarray(ids)


def stratified_split(X: pd.DataFrame, y: pd.Series, ids: np.ndarray,
                     seed: int = 42,
                     ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)
                     ) -> dict[str, tuple[pd.DataFrame, pd.Series, np.ndarray]]:
    """Stratified train/val/test split preserving the case rate."""
    rng = np.random.default_rng(seed)
    total = sum(ratios)
    split_idx: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for cls in (0, 1):
        cls_idx = np.where(y.to_numpy() == cls)[0].tolist()
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        n_train = int(round(n * ratios[0] / total))
        n_val = int(round(n * ratios[1] / total))
        split_idx["train"].extend(cls_idx[:n_train])
        split_idx["val"].extend(cls_idx[n_train:n_train + n_val])
        split_idx["test"].extend(cls_idx[n_train + n_val:])
    out: dict[str, tuple[pd.DataFrame, pd.Series, np.ndarray]] = {}
    for name in ("train", "val", "test"):
        take = np.array(sorted(split_idx[name]))
        out[name] = (X.iloc[take].reset_index(drop=True),
                     y.iloc[take].reset_index(drop=True),
                     ids[take])
    return out


#: Expected number of encoded features: 44 vars x 8 stats + n_hours.
N_FEATURES = len(FEATURE_VARS) * len(_STAT_NAMES) + 1
