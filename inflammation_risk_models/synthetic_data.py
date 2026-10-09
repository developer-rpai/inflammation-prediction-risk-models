"""Synthetic ICU-style time-series generator for inflammation prediction.

Produces hourly vital-sign and laboratory measurements for synthetic ICU stays,
following the labeling logic documented in the project README:

* **Suspicion of inflammatory response**: a culture sample and a clinical
  intervention (antibiotics) close together in time — sample first then
  treatment within 72 h, or treatment first then sample within 24 h.
* **Systemic dysfunction**: a simplified SOFA score increasing by >= 2 points
  over the patient's baseline.

The generated frame has one row per patient-hour and the same conceptual
schema the original MIMIC-III pipeline used: 44 clinical variables
(15 vitals + 29 labs), intervention flags, a SOFA score, an onset hour, and
a true_case flag marking patients that received the synthetic deterioration
(the label itself is a computable phenotype, so flag-free patients can
occasionally meet the criteria too).

Everything is synthetic: no real patient data, no credentials needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# (name, population mean, population std, plausible low, plausible high)
VITALS = [
    ("heart_rate", 78.0, 14.0, 30.0, 180.0),
    ("resp_rate", 18.0, 5.0, 8.0, 45.0),
    ("spo2", 97.0, 2.5, 70.0, 100.0),
    ("temp_c", 37.0, 0.7, 34.0, 41.5),
    ("sys_bp", 122.0, 20.0, 60.0, 230.0),
    ("dia_bp", 70.0, 12.0, 30.0, 140.0),
    ("map", 88.0, 14.0, 35.0, 170.0),
    ("fio2", 0.40, 0.18, 0.21, 1.0),
    ("glucose", 135.0, 55.0, 40.0, 600.0),
    ("urine_output", 55.0, 45.0, 0.0, 400.0),
    ("gcs", 14.0, 1.8, 3.0, 15.0),
    ("weight_kg", 80.0, 20.0, 35.0, 200.0),
    ("height_cm", 170.0, 10.0, 140.0, 205.0),
    ("cvp", 8.0, 4.5, 0.0, 30.0),
    ("peep", 5.0, 2.5, 0.0, 20.0),
]

LABS = [
    ("wbc", 10.0, 5.0, 0.5, 60.0),
    ("lactate", 1.8, 1.2, 0.3, 20.0),
    ("creatinine", 1.1, 0.8, 0.2, 12.0),
    ("platelets", 220.0, 90.0, 10.0, 700.0),
    ("bilirubin_total", 1.0, 1.2, 0.1, 25.0),
    ("bun", 22.0, 15.0, 3.0, 150.0),
    ("sodium", 139.0, 5.0, 110.0, 165.0),
    ("potassium", 4.1, 0.7, 2.0, 7.5),
    ("chloride", 103.0, 6.0, 75.0, 130.0),
    ("bicarbonate", 24.0, 5.0, 8.0, 45.0),
    ("hemoglobin", 11.0, 2.2, 4.0, 20.0),
    ("hematocrit", 33.0, 6.5, 12.0, 60.0),
    ("alt", 45.0, 60.0, 5.0, 2000.0),
    ("ast", 55.0, 80.0, 8.0, 3000.0),
    ("alp", 90.0, 60.0, 20.0, 800.0),
    ("albumin", 3.4, 0.7, 1.0, 5.5),
    ("calcium", 8.6, 0.9, 5.0, 12.0),
    ("magnesium", 2.0, 0.4, 0.8, 4.5),
    ("phosphate", 3.5, 1.2, 1.0, 10.0),
    ("ptt", 32.0, 12.0, 20.0, 150.0),
    ("inr", 1.2, 0.5, 0.8, 8.0),
    ("fibrinogen", 380.0, 150.0, 80.0, 900.0),
    ("crp", 60.0, 80.0, 0.5, 400.0),
    ("procalcitonin", 1.5, 3.0, 0.02, 60.0),
    ("troponin", 0.2, 0.8, 0.0, 25.0),
    ("ck", 250.0, 500.0, 10.0, 8000.0),
    ("ldh", 280.0, 200.0, 80.0, 3000.0),
    ("d_dimer", 1.2, 2.0, 0.1, 20.0),
    ("anion_gap", 13.0, 5.0, 2.0, 35.0),
]

#: All 44 clinical variable names, vitals first then labs.
FEATURE_VARS = [name for name, _, _, _, _ in VITALS + LABS]

#: Multiplicative deterioration applied at full ramp (progress = 1.0).
_RAMP_MULT = {
    "heart_rate": 1.35, "resp_rate": 1.45, "wbc": 2.3, "lactate": 2.8,
    "crp": 3.0, "procalcitonin": 3.5, "creatinine": 1.6, "bilirubin_total": 1.9,
    "bun": 1.5, "glucose": 1.25, "d_dimer": 2.0, "anion_gap": 1.35,
    "map": 0.75, "platelets": 0.65, "fibrinogen": 0.8, "bicarbonate": 0.83,
}
#: Additive deterioration applied at full ramp.
_RAMP_ADD = {
    "temp_c": 1.6, "spo2": -4.0, "gcs": -2.5,
}


def _ar1(rng: np.random.Generator, n: int, mean: float, std: float,
         lo: float, hi: float, rho: float = 0.85) -> np.ndarray:
    """Draw a temporally correlated hourly series, clipped to a plausible range.

    Uses a hierarchical structure: a patient-level offset (between-patient
    variation) plus AR(1) hourly noise (within-patient variation), so a
    patient's own trajectory is smoother than the population spread --
    as in real ICU chart data.
    """
    offset = rng.normal(0.0, 0.8 * std)
    mu = mean + offset
    wstd = 0.6 * std
    eps = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = mu + wstd * eps[0]
    for t in range(1, n):
        x[t] = mu + rho * (x[t - 1] - mu) + wstd * np.sqrt(1 - rho ** 2) * eps[t]
    return np.clip(x, lo, hi)


def _sofa_proxy(df: pd.DataFrame) -> np.ndarray:
    """Simplified SOFA score (0-24) from generated vitals/labs.

    Uses standard SOFA cutoffs with SpO2/FiO2 as the PaO2/FiO2 proxy and
    MAP < 70 as the cardiovascular criterion (no vasopressor modeling).
    """
    n = len(df)
    sofa = np.zeros(n, dtype=float)

    ratio = df["spo2"].to_numpy() / df["fio2"].to_numpy()
    sofa += np.select(
        [ratio < 100, ratio < 200, ratio < 300, ratio < 400],
        [4, 3, 2, 1], default=0,
    )
    plt = df["platelets"].to_numpy()
    sofa += np.select(
        [plt < 20, plt < 50, plt < 100, plt < 150], [4, 3, 2, 1], default=0
    )
    bili = df["bilirubin_total"].to_numpy()
    sofa += np.select(
        [bili >= 12, bili >= 6, bili >= 2, bili >= 1.2], [4, 3, 2, 1], default=0
    )
    sofa += np.where(df["map"].to_numpy() < 70, 1, 0)
    gcs = df["gcs"].to_numpy()
    sofa += np.select(
        [gcs < 6, gcs < 10, gcs < 13, gcs < 15], [4, 3, 2, 1], default=0
    )
    creat = df["creatinine"].to_numpy()
    sofa += np.select(
        [creat >= 5, creat >= 3.5, creat >= 2, creat >= 1.2], [4, 3, 2, 1], default=0
    )
    return sofa


def _apply_deterioration(values: dict[str, np.ndarray], t0: int) -> None:
    """Ramp vitals/labs toward an inflammatory state around onset hour t0.

    The ramp starts 60 h before t0 and peaks 12 h after; progress is
    square-root shaped so early deterioration is already measurable --
    otherwise a 7 h prediction horizon would see only baseline physiology.
    """
    n = len(next(iter(values.values())))
    start, peak = t0 - 60, t0 + 12
    span = max(1, peak - start)
    for t in range(max(0, start), min(n, peak)):
        p = (t - start) / span
        q = p ** 0.5  # faster early rise
        for var, mult in _RAMP_MULT.items():
            values[var][t] *= 1 + (mult - 1) * q
        for var, add in _RAMP_ADD.items():
            values[var][t] += add * q


def generate_cohort(n_patients: int = 600, seed: int = 42,
                    case_rate: float = 0.12,
                    missing_rate: float = 0.15) -> pd.DataFrame:
    """Generate a synthetic ICU cohort as a long (patient-hour) DataFrame.

    Columns: patient_id, hour, the 44 clinical variables, culture_sampled,
    antibiotic_given, sofa, onset_hour (patient-level; NaN for controls).
    """
    rng = np.random.default_rng(seed)
    frames: list[pd.DataFrame] = []

    for pid in range(n_patients):
        n_hours = int(rng.integers(160, 281))
        is_case = rng.random() < case_rate

        values = {
            name: _ar1(rng, n_hours, mean, std, lo, hi)
            for name, mean, std, lo, hi in VITALS + LABS
        }

        culture_time: int | None = None
        abx_time: int | None = None
        if is_case:
            # t0 >= 108 guarantees the detectable onset stays >= 48 h, so the
            # early-prediction framing (predict from a 48 h window, 7 h
            # horizon) holds. t0 <= n_hours - 25 keeps the ramp in-stay.
            t0 = int(rng.integers(108, n_hours - 24))
            _apply_deterioration(values, t0)
            # Re-clip after deterioration so values stay plausible.
            for (name, mean, std, lo, hi) in VITALS + LABS:
                values[name] = np.clip(values[name], lo, hi)
            culture_time, abx_time = t0 - 18, t0 - 6
        elif rng.random() < 0.02:
            # A few controls get an unrelated culture/antibiotic pair, placed
            # after hour 48 so any resulting label stays in the predictable
            # region (early-prediction framing: onset >= 48 h).
            culture_time = int(rng.integers(50, n_hours - 48))
            abx_time = min(culture_time + int(rng.integers(0, 48)),
                           n_hours - 1)

        df = pd.DataFrame(values)
        df["culture_sampled"] = 0
        df["antibiotic_given"] = 0
        if culture_time is not None:
            df.loc[culture_time, "culture_sampled"] = 1
        if abx_time is not None:
            df.loc[abx_time, "antibiotic_given"] = 1

        df["sofa"] = _sofa_proxy(df)
        baseline = df["sofa"].iloc[:24].min()
        dysfunction = (df["sofa"] - baseline) >= 2

        hours = np.arange(n_hours)
        suspicion = np.zeros(n_hours, dtype=bool)
        if culture_time is not None and abx_time is not None:
            if abx_time >= culture_time:
                # Sample first: treatment must follow within 72 h.
                window = (hours >= culture_time) & (hours <= abx_time + 72)
            else:
                # Treatment first: sample must follow within 24 h.
                window = (hours >= abx_time) & (hours <= culture_time + 24)
            suspicion |= window

        onset_idx = np.where(suspicion & dysfunction.to_numpy())[0]
        onset_hour = float(onset_idx[0]) if len(onset_idx) else np.nan

        # MCAR missingness on the raw measurements only (flags/scores stay).
        mask = rng.random((n_hours, len(FEATURE_VARS))) < missing_rate
        df[FEATURE_VARS] = df[FEATURE_VARS].mask(mask)

        df.insert(0, "hour", hours)
        df.insert(0, "patient_id", pid)
        df["onset_hour"] = onset_hour
        # Generative flag: did this patient receive the deterioration ramp?
        # (The label is a computable phenotype; flag-free controls can still
        # meet the criteria, so true_case != (onset is not None) in general.)
        df["true_case"] = int(is_case)
        frames.append(df)

    cohort = pd.concat(frames, ignore_index=True)
    cols = (["patient_id", "hour"] + FEATURE_VARS
            + ["culture_sampled", "antibiotic_given", "sofa", "onset_hour",
               "true_case"])
    return cohort[cols]


if __name__ == "__main__":
    cohort = generate_cohort(n_patients=50, seed=0)
    cases = cohort.drop_duplicates("patient_id")["onset_hour"].notna().mean()
    print(f"patients={cohort['patient_id'].nunique()} "
          f"rows={len(cohort)} case_rate={cases:.3f}")
    print(cohort.head(3).to_string())
