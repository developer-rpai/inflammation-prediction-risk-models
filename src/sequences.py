"""Raw-sequence dataset builder for the recurrent (GRU) modeling path.

Mirrors the patient-selection and prediction-time logic in
:mod:`src.preprocessing` (same ``OBS_WINDOW``, ``PREDICTION_HORIZON`` and
``FOLLOWUP_WINDOW`` exclusion rules), but instead of collapsing each 48-hour
observation window into summary statistics, it returns the raw hourly
sequences so recurrent models can learn temporal patterns directly.

Imputation follows the scheme documented in the README for recurrent models:
forward-fill within the patient's stay, then the variable's training-set
median. Standardization (z-score) uses training-set statistics only, so no
information leaks from validation/test into the preprocessing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocessing import FOLLOWUP_WINDOW, OBS_WINDOW, PREDICTION_HORIZON
from src.synthetic_data import FEATURE_VARS


def _window_frame(grp: pd.DataFrame, end_hour: int,
                  obs_window: int = OBS_WINDOW) -> pd.DataFrame:
    """Return the ``obs_window`` hourly rows ending at ``end_hour``.

    The frame is reindexed to the complete hour range so every window has
    exactly ``obs_window`` rows; missing hours stay NaN for the imputation
    step to handle.
    """
    hours = np.arange(end_hour - obs_window, end_hour)
    win = (grp.set_index("hour")
              .reindex(hours)[FEATURE_VARS]
              .reset_index(drop=True))
    return win


def _prediction_times(cohort: pd.DataFrame, seed: int = 42,
                      obs_window: int = OBS_WINDOW,
                      horizon: int = PREDICTION_HORIZON,
                      followup: int = FOLLOWUP_WINDOW):
    """Yield ``(patient_group, prediction_hour, label)`` per eligible patient.

    Identical eligibility rules to :func:`src.preprocessing.build_dataset`:
    cases are predicted ``horizon`` hours before onset (excluded when the
    onset is too early for a full window); controls are predicted at a random
    hour with a full history and an onset-free follow-up.
    """
    rng = np.random.default_rng(seed)
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
        yield grp, pred_time, label, pid


def build_sequences(cohort: pd.DataFrame, seed: int = 42,
                    obs_window: int = OBS_WINDOW,
                    horizon: int = PREDICTION_HORIZON,
                    followup: int = FOLLOWUP_WINDOW
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the raw-sequence modeling dataset.

    Returns ``(X, y, ids)`` with ``X`` shaped
    ``(n_patients, obs_window, n_features)``. Values are raw (NaN where the
    stay has no measurement); call :func:`impute_and_standardize` before
    training.
    """
    seqs, labels, ids = [], [], []
    for grp, pred_time, label, pid in _prediction_times(
            cohort, seed, obs_window, horizon, followup):
        seqs.append(_window_frame(grp, pred_time, obs_window).to_numpy(
            dtype=float))
        labels.append(label)
        ids.append(pid)
    X = np.stack(seqs).astype(np.float32)
    return X, np.asarray(labels, dtype=np.int64), np.asarray(ids)


def impute_and_standardize(
        X_train: np.ndarray, *others: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Forward-fill, median-fill and z-score sequences.

    Medians, means and standard deviations are computed on ``X_train`` only
    and applied to every array, so validation/test statistics never leak
    into preprocessing. After standardization any residual NaN becomes 0
    (the standardized mean).
    """
    def _ffill(a: np.ndarray) -> np.ndarray:
        out = a.copy()
        # forward fill along the time axis, per patient and variable
        mask = np.isnan(out)
        # prepend a NaN row so leading NaNs stay NaN for the median step
        for i in range(out.shape[0]):
            df = pd.DataFrame(out[i])
            out[i] = df.ffill().to_numpy()
        return out

    tr = _ffill(X_train)
    medians = np.nanmedian(tr.reshape(-1, tr.shape[2]), axis=0)
    means = np.nanmean(tr.reshape(-1, tr.shape[2]), axis=0)
    stds = np.nanstd(tr.reshape(-1, tr.shape[2]), axis=0)
    stds[stds == 0] = 1.0

    def _transform(a: np.ndarray) -> np.ndarray:
        b = _ffill(a)
        for j in range(b.shape[2]):
            col = b[:, :, j]
            col[np.isnan(col)] = medians[j]
            b[:, :, j] = (col - means[j]) / stds[j]
        return np.nan_to_num(b, nan=0.0).astype(np.float32)

    return (_transform(tr),) + tuple(_transform(o) for o in others)


def sequence_split(X: np.ndarray, y: np.ndarray, ids: np.ndarray,
                   seed: int = 42,
                   ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)
                   ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Stratified train/val/test split for sequence arrays.

    Same stratification logic as :func:`src.preprocessing.stratified_split`,
    adapted to ndarrays (the DataFrame version uses ``.iloc``, which arrays
    don't have).
    """
    rng = np.random.default_rng(seed)
    total = sum(ratios)
    split_idx: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for cls in (0, 1):
        cls_idx = np.where(y == cls)[0].tolist()
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        n_train = int(round(n * ratios[0] / total))
        n_val = int(round(n * ratios[1] / total))
        split_idx["train"].extend(cls_idx[:n_train])
        split_idx["val"].extend(cls_idx[n_train:n_train + n_val])
        split_idx["test"].extend(cls_idx[n_train + n_val:])
    out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name in ("train", "val", "test"):
        take = np.array(sorted(split_idx[name]))
        out[name] = (X[take], y[take], ids[take])
    return out
