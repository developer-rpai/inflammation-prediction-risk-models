#!/usr/bin/env python3
"""Seeded generator for synthetic MIMIC-shaped tables.

Every value is drawn from ``numpy.random.Generator``. Rows are not copies or
perturbations of real MIMIC-III stays. Identifiers start at 900000 (subjects),
910000 (hospital admissions), and 920000 (ICU stays), outside the MIMIC-III
icustay_id range.

Run from the repository root::

    python scripts/generate_synthetic_mimic.py

Outputs land in ``data/synthetic/`` with a ``SYNTHETIC_`` filename prefix.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "synthetic"

# Obviously fake identifier ranges. MIMIC-III icustay_id values sit in 2xxxxx.
SUBJECT_ID_START = 900_000
HADM_ID_START = 910_000
ICUSTAY_ID_START = 920_000
EXAMPLE_ICUSTAY_ID = ICUSTAY_ID_START  # first generated stay; notebooks filter on this

# Same array lengths as the historical split pickles, filled with fake ids.
SPLIT_TRAIN = 4965
SPLIT_VAL = 620
SPLIT_TEST = 620
N_SPLIT_IDS = SPLIT_TRAIN + SPLIT_VAL + SPLIT_TEST  # 6205

N_SUBJECTS = 80
EXTRA_STAY_FRACTION = 0.25
N_CASES = 24
N_CONTROLS = 48
HOURS = 48

GENDERS = ["F", "M"]
# Public MIMIC-III codebook labels (schema vocabulary, not patient rows).
ETHNICITIES = [
    "WHITE",
    "BLACK/AFRICAN AMERICAN",
    "HISPANIC OR LATINO",
    "ASIAN",
    "OTHER",
    "UNKNOWN/NOT SPECIFIED",
    "UNABLE TO OBTAIN",
    "AMERICAN INDIAN/ALASKA NATIVE",
]
ADMISSION_TYPES = ["EMERGENCY", "ELECTIVE", "URGENT"]
ADMISSION_LOCATIONS = [
    "EMERGENCY ROOM ADMIT",
    "PHYS REFERRAL/NORMAL DELI",
    "CLINIC REFERRAL/PREMATURE",
    "TRANSFER FROM HOSP/EXTRAM",
]
CARE_UNITS = ["MICU", "SICU", "CCU", "CSRU", "TSICU"]

STATIC_COLUMNS = [
    "subject_id",
    "hadm_id",
    "icustay_id",
    "gender",
    "admittime",
    "admission_age",
    "ethnicity",
    "admission_type",
    "admission_location",
    "first_careunit",
    "hospstay_seq",
    "first_hosp_stay",
    "intime",
    "outtime",
    "icustay_seq",
    "first_icu_stay",
]

# (name, mean, std, low, high, observation probability)
VITALS = [
    ("sysbp", 120.0, 18.0, 60.0, 220.0, 0.70),
    ("diabp", 68.0, 12.0, 30.0, 130.0, 0.70),
    ("meanbp", 82.0, 12.0, 40.0, 150.0, 0.70),
    ("resprate", 18.0, 5.0, 8.0, 40.0, 0.75),
    ("heartrate", 88.0, 16.0, 40.0, 160.0, 0.75),
    ("spo2_pulsoxy", 97.0, 2.0, 80.0, 100.0, 0.72),
    ("tempc", 37.0, 0.6, 35.0, 40.5, 0.35),
    ("cardiacoutput", 5.2, 1.4, 1.5, 12.0, 0.08),
    ("sv", 70.0, 15.0, 20.0, 140.0, 0.06),
    ("svi", 38.0, 8.0, 15.0, 70.0, 0.05),
    ("svv", 12.0, 4.0, 2.0, 30.0, 0.05),
    ("tfc", 35.0, 8.0, 10.0, 70.0, 0.05),
    ("tpr", 1100.0, 250.0, 400.0, 2500.0, 0.05),
    ("tvset", 480.0, 60.0, 250.0, 750.0, 0.12),
    ("tvobserved", 470.0, 70.0, 200.0, 800.0, 0.12),
    ("tvspontaneous", 350.0, 80.0, 50.0, 700.0, 0.06),
    ("flowrate", 40.0, 15.0, 5.0, 80.0, 0.08),
    ("peakinsppressure", 22.0, 6.0, 8.0, 45.0, 0.12),
    ("totalpeeplevel", 6.0, 2.0, 0.0, 18.0, 0.08),
    ("vitalcapacity", 2.4, 0.7, 0.4, 5.0, 0.04),
    ("o2flow", 4.0, 2.0, 0.0, 15.0, 0.15),
    ("fio2", 0.40, 0.15, 0.21, 1.0, 0.18),
    ("crp", 80.0, 60.0, 1.0, 400.0, 0.08),
]
LABS = [
    ("aniongap", 13.0, 4.0, 3.0, 30.0, 0.20),
    ("albumin", 3.2, 0.6, 1.2, 5.2, 0.08),
    ("bands", 4.0, 3.0, 0.0, 30.0, 0.03),
    ("bicarbonate", 24.0, 4.0, 10.0, 40.0, 0.22),
    ("bilirubin", 1.1, 1.0, 0.1, 15.0, 0.08),
    ("creatinine", 1.2, 0.8, 0.3, 8.0, 0.22),
    ("chloride", 104.0, 5.0, 85.0, 125.0, 0.25),
    ("glucose", 140.0, 45.0, 50.0, 400.0, 0.28),
    ("hematocrit", 32.0, 5.0, 18.0, 52.0, 0.28),
    ("hemoglobin", 10.5, 1.8, 6.0, 17.0, 0.25),
    ("lactate", 1.8, 1.0, 0.4, 12.0, 0.15),
    ("platelet", 210.0, 80.0, 20.0, 600.0, 0.22),
    ("potassium", 4.1, 0.5, 2.5, 6.5, 0.30),
    ("ptt", 34.0, 10.0, 20.0, 120.0, 0.15),
    ("inr", 1.3, 0.4, 0.8, 6.0, 0.15),
    ("pt", 14.0, 3.0, 9.0, 40.0, 0.15),
    ("sodium", 139.0, 4.0, 120.0, 160.0, 0.25),
    ("bun", 24.0, 12.0, 5.0, 120.0, 0.22),
    ("wbc", 11.0, 5.0, 1.0, 40.0, 0.22),
    ("ferritin", 400.0, 300.0, 20.0, 3000.0, 0.04),
    ("transferrin", 200.0, 50.0, 80.0, 400.0, 0.03),
    ("creatinekinase", 250.0, 200.0, 20.0, 4000.0, 0.08),
    ("ck_mb", 5.0, 4.0, 0.5, 40.0, 0.07),
    ("d_dimer", 1.5, 1.2, 0.1, 15.0, 0.04),
    ("ntprobnp", 800.0, 700.0, 20.0, 8000.0, 0.03),
    ("sedimentationrate", 30.0, 20.0, 2.0, 120.0, 0.03),
    ("fibrinogen", 350.0, 120.0, 80.0, 800.0, 0.06),
    ("ldh", 280.0, 150.0, 80.0, 1500.0, 0.06),
    ("magnesium", 2.0, 0.3, 1.0, 3.5, 0.18),
    ("calcium_free", 1.15, 0.1, 0.8, 1.6, 0.12),
    ("po2_bloodgas", 90.0, 25.0, 40.0, 250.0, 0.12),
    ("ph_bloodgas", 7.38, 0.06, 7.05, 7.60, 0.12),
    ("pco2_bloodgas", 40.0, 8.0, 20.0, 80.0, 0.12),
    ("so2_bloodgas", 96.0, 3.0, 70.0, 100.0, 0.08),
    ("troponin_t", 0.08, 0.1, 0.0, 2.0, 0.05),
    ("glucose_csf", 70.0, 15.0, 30.0, 150.0, 0.01),
    ("total_protein_joint_fluid", 3.5, 1.0, 1.0, 8.0, 0.01),
    ("total_protein_pleural", 3.2, 1.0, 0.5, 8.0, 0.01),
    ("urine_albumin_creatinine_ratio", 40.0, 30.0, 1.0, 300.0, 0.01),
    ("wbc_ascites", 200.0, 150.0, 0.0, 2000.0, 0.01),
]

# Columns of the shorter modeling table (49 columns total, including keys).
DROPPED_SHORT_VITALS = [
    "sysbp", "diabp", "meanbp", "resprate", "heartrate", "spo2_pulsoxy",
    "tempc", "cardiacoutput", "tvset", "tvobserved", "tvspontaneous",
    "peakinsppressure", "totalpeeplevel", "o2flow", "fio2",
]
DROPPED_SHORT_LABS = [
    "albumin", "bands", "bicarbonate", "bilirubin", "creatinine", "chloride",
    "glucose", "hematocrit", "hemoglobin", "lactate", "platelet", "potassium",
    "ptt", "inr", "pt", "sodium", "bun", "wbc", "creatinekinase", "ck_mb",
    "fibrinogen", "ldh", "magnesium", "calcium_free", "po2_bloodgas",
    "ph_bloodgas", "pco2_bloodgas", "so2_bloodgas", "troponin_t",
]
BINNED_FEATURES = DROPPED_SHORT_VITALS + DROPPED_SHORT_LABS


def _ts(rng: np.random.Generator, origin: np.datetime64, max_hours: int) -> pd.Timestamp:
    hours = int(rng.integers(0, max_hours))
    return pd.Timestamp(origin) + pd.Timedelta(hours=hours)


def _series(rng: np.random.Generator, n: int, mean: float, std: float,
            lo: float, hi: float, p_obs: float) -> np.ndarray:
    values = np.clip(rng.normal(mean, std, size=n), lo, hi).astype(float)
    missing = rng.random(n) > p_obs
    values[missing] = np.nan
    return values


def build_stays(rng: np.random.Generator) -> pd.DataFrame:
    """One row per synthetic ICU stay, with case/control flags."""
    extra = rng.random(N_SUBJECTS) < EXTRA_STAY_FRACTION
    rows = []
    stay_i = 0
    origin = np.datetime64("2000-01-01")
    for subject_i in range(N_SUBJECTS):
        n_stays = 2 if extra[subject_i] else 1
        subject_id = SUBJECT_ID_START + subject_i
        for seq in range(1, n_stays + 1):
            admittime = _ts(rng, origin, 24 * 400)
            delay_h = int(rng.integers(1, 36))
            los_h = float(rng.uniform(24.0, 180.0))
            intime = admittime + pd.Timedelta(hours=delay_h)
            outtime = intime + pd.Timedelta(hours=los_h)
            onset_hour = float(rng.integers(8, 36))
            rows.append({
                "subject_id": subject_id,
                "hadm_id": HADM_ID_START + stay_i,
                "icustay_id": ICUSTAY_ID_START + stay_i,
                "gender": str(rng.choice(GENDERS)),
                "admittime": admittime,
                "admission_age": float(rng.uniform(18.0, 90.0)),
                "ethnicity": str(rng.choice(ETHNICITIES)),
                "admission_type": str(rng.choice(ADMISSION_TYPES)),
                "admission_location": str(rng.choice(ADMISSION_LOCATIONS)),
                "first_careunit": str(rng.choice(CARE_UNITS)),
                "hospstay_seq": seq,
                "first_hosp_stay": int(seq == 1),
                "intime": intime,
                "outtime": outtime,
                "icustay_seq": seq,
                "first_icu_stay": int(seq == 1),
                "length_of_stay": los_h,
                "onset_hour": onset_hour,
            })
            stay_i += 1
    stays = pd.DataFrame(rows)
    # One missing discharge time, matching the nullable outtime in the schema.
    stays.loc[stays.index[-1], "outtime"] = pd.NaT
    stays.loc[stays.index[-1], "length_of_stay"] = np.nan

    case_ids = stays["icustay_id"].iloc[:N_CASES].to_numpy()
    control_ids = stays["icustay_id"].iloc[N_CASES:N_CASES + N_CONTROLS].to_numpy()
    stays["is_case"] = stays["icustay_id"].isin(case_ids)
    stays["is_control"] = stays["icustay_id"].isin(control_ids)
    matched = rng.choice(case_ids, size=len(control_ids), replace=True)
    stays["matched_case_icustay_id"] = pd.Series(pd.NA, index=stays.index, dtype="Int64")
    stays.loc[stays["is_control"], "matched_case_icustay_id"] = matched
    # The example id used by the time-series notebook is a case.
    if int(stays.iloc[0]["icustay_id"]) != EXAMPLE_ICUSTAY_ID or not bool(stays.iloc[0]["is_case"]):
        raise RuntimeError("example icustay is not the first case stay")
    return stays


def static_frames(stays: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    static = stays[STATIC_COLUMNS].copy()

    cases = stays.loc[stays["is_case"]].copy()
    cases["icustay_id.1"] = cases["icustay_id"]
    cases["intime.1"] = cases["intime"]
    cases["outtime.1"] = cases["outtime"]
    cases["delta_score"] = 2 + (cases["icustay_id"] % 5)
    cases["inflammation_onset"] = cases["intime"] + pd.to_timedelta(cases["onset_hour"], unit="h")
    cases["inflammation_onset_day"] = np.floor(cases["onset_hour"] / 24.0)
    cases["inflammation_onset_hour"] = cases["onset_hour"]
    case_cols = STATIC_COLUMNS + [
        "icustay_id.1", "intime.1", "outtime.1", "length_of_stay", "delta_score",
        "inflammation_onset", "inflammation_onset_day", "inflammation_onset_hour",
    ]
    cases = cases[case_cols]

    controls = stays.loc[stays["is_control"]].copy()
    controls["icustay_id.1"] = controls["icustay_id"]
    controls["hadm_id.1"] = controls["hadm_id"]
    controls["intime.1"] = controls["intime"]
    controls["outtime.1"] = controls["outtime"]
    controls["control_onset_hour"] = controls["onset_hour"]
    controls["control_onset_time"] = controls["intime"] + pd.to_timedelta(controls["onset_hour"], unit="h")
    control_cols = STATIC_COLUMNS + [
        "icustay_id.1", "hadm_id.1", "intime.1", "outtime.1", "length_of_stay",
        "control_onset_hour", "control_onset_time", "matched_case_icustay_id",
    ]
    controls = controls[control_cols]
    return static, cases, controls


def _hourly_block(rng: np.random.Generator, stays: pd.DataFrame, specs, target_name: str) -> pd.DataFrame:
    frames = []
    for row in stays.itertuples(index=False):
        hours = np.arange(HOURS)
        frame = pd.DataFrame({
            "icustay_id": int(row.icustay_id),
            "subject_id": int(row.subject_id),
            "chart_time": pd.Timestamp(row.intime) + pd.to_timedelta(hours, unit="h"),
            target_name: (hours >= row.onset_hour).astype(int),
        })
        for name, mean, std, lo, hi, p_obs in specs:
            frame[name] = _series(rng, HOURS, mean, std, lo, hi, p_obs)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def hourly_tables(rng: np.random.Generator, stays: pd.DataFrame):
    cases = stays.loc[stays["is_case"]]
    controls = stays.loc[stays["is_control"]]
    case_labs = _hourly_block(rng, cases, LABS, "inflammation_target")
    control_labs = _hourly_block(rng, controls, LABS, "pseudo_target")
    case_vitals = _hourly_block(rng, cases, VITALS, "inflammation_target")
    control_vitals = _hourly_block(rng, controls, VITALS, "pseudo_target")
    return case_labs, control_labs, case_vitals, control_vitals


def full_labvitals(rng: np.random.Generator, stays: pd.DataFrame) -> pd.DataFrame:
    """Wide and short modeling tables keyed by stay and hour offset."""
    cohort = stays.loc[stays["is_case"] | stays["is_control"]]
    frames = []
    spec_map = {name: (mean, std, lo, hi, p) for name, mean, std, lo, hi, p in VITALS + LABS}
    for row in cohort.itertuples(index=False):
        hours = np.arange(HOURS, dtype=float)
        is_case = bool(row.is_case)
        frame = pd.DataFrame({
            "label": ((hours >= row.onset_hour) & is_case).astype(int),
            "icustay_id": int(row.icustay_id),
            "subject_id": int(row.subject_id),
            "chart_time": hours,
            "inflammation_target": ((hours >= row.onset_hour) & is_case).astype(int),
        })
        for name, (mean, std, lo, hi, p_obs) in spec_map.items():
            frame[name] = _series(rng, HOURS, mean, std, lo, hi, p_obs)
        frames.append(frame)
    wide_cols = (
        ["label", "icustay_id", "subject_id", "chart_time", "inflammation_target"]
        + [name for name, *_ in VITALS]
        + [name for name, *_ in LABS]
    )
    wide = pd.concat(frames, ignore_index=True)[wide_cols]
    short_cols = (
        ["label", "icustay_id", "subject_id", "chart_time", "inflammation_target"]
        + DROPPED_SHORT_VITALS
        + DROPPED_SHORT_LABS
    )
    return wide, wide[short_cols].copy()


def q13_and_sofa(rng: np.random.Generator, stays: pd.DataFrame, cases: pd.DataFrame, controls: pd.DataFrame):
    q13_cases = cases[[
        "icustay_id", "intime", "outtime", "length_of_stay", "delta_score",
        "inflammation_onset", "inflammation_onset_day", "inflammation_onset_hour",
    ]].copy()
    q13_controls = controls[[
        "icustay_id", "hadm_id", "intime", "outtime", "length_of_stay",
        "control_onset_hour", "control_onset_time", "matched_case_icustay_id",
    ]].copy()
    # Stay-level table with the columns the preview notebook expects.
    other = stays.loc[~stays["is_case"]].copy()
    other["delta_score"] = (other["icustay_id"] % 4).astype(int)
    other["inflammation_onset"] = pd.NaT
    q13_controls_hourly = other[[
        "icustay_id", "hadm_id", "intime", "outtime", "length_of_stay",
        "delta_score", "inflammation_onset",
    ]].copy()

    sofa_rows = []
    for row in stays.itertuples(index=False):
        for window in range(12):
            parts = rng.integers(0, 5, size=6).astype(float)
            if rng.random() < 0.3:
                parts[int(rng.integers(0, 6))] = np.nan
            sofa_rows.append({
                "icustay_id": int(row.icustay_id),
                "sofa": float(np.nansum(parts)),
                "time_window": window,
                "respiration": parts[0],
                "coagulation": parts[1],
                "liver": parts[2],
                "cardiovascular": parts[3],
                "cns": parts[4],
                "renal": parts[5],
                "cv_mean_bp_u70": float(rng.integers(0, 2)),
                "cv_vasopressor_usage": float(rng.integers(0, 2)),
            })
    sofa = pd.DataFrame(sofa_rows)

    delta_rows = []
    for row in stays.itertuples(index=False):
        comps = rng.integers(0, 5, size=6)
        delta_rows.append([
            int(row.icustay_id),
            int(rng.integers(2, 12)),
            pd.Timestamp(row.intime) + pd.Timedelta(hours=int(row.onset_hour)),
            float(comps[0]), float(comps[1]), float(comps[2]),
            int(comps.sum()),
            float(comps[3]), float(comps[4]),
            int(rng.integers(0, 2)), int(rng.integers(0, 2)),
        ])
    sofa_delta = pd.DataFrame(delta_rows)

    si = pd.DataFrame({
        "icustay_id": stays["icustay_id"].astype(int),
        "si_starttime": [
            (pd.Timestamp(t) + pd.Timedelta(hours=int(h))) if rng.random() > 0.4 else pd.NaT
            for t, h in zip(stays["intime"], stays["onset_hour"])
        ],
    })
    return q13_cases, q13_controls, q13_controls_hourly, sofa, sofa_delta, si


def split_pickles(rng: np.random.Generator) -> list[dict]:
    """Three reshuffles of the same fake icustay id pool."""
    pool = np.arange(ICUSTAY_ID_START, ICUSTAY_ID_START + N_SPLIT_IDS, dtype=np.int64)
    splits = []
    for _ in range(3):
        perm = rng.permutation(pool)
        splits.append({
            "train": perm[:SPLIT_TRAIN],
            "validation": perm[SPLIT_TRAIN:SPLIT_TRAIN + SPLIT_VAL],
            "test": perm[SPLIT_TRAIN + SPLIT_VAL:],
        })
    return splits


def _feature_split(rng: np.random.Generator, n_stays: int = 6) -> list:
    """Per-feature object arrays of short random series (binned-pickle shape)."""
    columns = []
    for _ in BINNED_FEATURES:
        series = [
            rng.normal(0.0, 1.0, size=int(rng.integers(8, 16))).astype(float)
            for _ in range(n_stays)
        ]
        columns.append(np.array(series, dtype=object))
    return columns


def binned_objects(rng: np.random.Generator):
    names = np.array(BINNED_FEATURES, dtype="<U16")
    train, val, test = (_feature_split(rng) for _ in range(3))
    compact = [names, train, val, test]
    labels = [
        rng.integers(0, 2, size=6).astype(np.int64),
        rng.integers(0, 2, size=6).astype(np.int64),
        rng.integers(0, 2, size=6).astype(np.int64),
    ]
    mgp = [names, train, val, test, *labels]
    return compact, mgp


def _write_csv(df: pd.DataFrame, path: Path, header: bool = True) -> None:
    df.to_csv(path, index=False, header=header)


def generate(seed: int = SEED, out_dir: Path = OUT_DIR) -> Path:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    stays = build_stays(rng)
    static, cases, controls = static_frames(stays)
    case_labs, control_labs, case_vitals, control_vitals = hourly_tables(rng, stays)
    wide, short = full_labvitals(rng, stays)
    q13_cases, q13_matched, q13_controls_hourly, sofa, sofa_delta, si = q13_and_sofa(
        rng, stays, cases, controls
    )
    splits = split_pickles(rng)
    compact, mgp = binned_objects(rng)

    files = {
        "SYNTHETIC_static_variables.csv": static,
        "SYNTHETIC_static_variables_cases.csv": cases,
        "SYNTHETIC_static_variables_controls.csv": controls,
        "SYNTHETIC_case_55h_hourly_labs_ex1c.csv": case_labs,
        "SYNTHETIC_control_55h_hourly_labs_ex1c.csv": control_labs,
        "SYNTHETIC_case_55h_hourly_vitals_ex1c.csv": case_vitals,
        "SYNTHETIC_control_55h_hourly_vitals_ex1c.csv": control_vitals,
        "SYNTHETIC_case_55h_hourly_labs_ex1c_collected.csv": case_labs,
        "SYNTHETIC_control_55h_hourly_labs_ex1c_collected.csv": control_labs,
        "SYNTHETIC_q13_cases_hourly_ex1c.csv": q13_cases,
        "SYNTHETIC_q13_matched_controls.csv": q13_matched,
        "SYNTHETIC_q13_controls_hourly.csv": q13_controls_hourly,
        "SYNTHETIC_sofa_table.csv": sofa,
        "SYNTHETIC_si_starttime.csv": si,
        "SYNTHETIC_full_labvitals_horizon_0.csv": wide,
        "SYNTHETIC_full_labvitals_horizon_0_dropped_short.csv": short,
    }
    for name, frame in files.items():
        _write_csv(frame, out_dir / name)
    _write_csv(sofa_delta, out_dir / "SYNTHETIC_sofa_delta.csv", header=False)

    for i, split in enumerate(splits):
        with open(out_dir / f"SYNTHETIC_tvt_info_split_{i}.pkl", "wb") as fh:
            pickle.dump(split, fh)
    with open(out_dir / "SYNTHETIC_labvitals_tr_te_val_binned_min_length_7_max_length_200_horizon_0_split_0.pkl", "wb") as fh:
        pickle.dump(compact, fh)
    with open(out_dir / "SYNTHETIC_labvitals_tr_te_val_compact_min_length_None_max_length_None_horizon_0_split_0.pkl", "wb") as fh:
        pickle.dump(compact, fh)
    with open(out_dir / "SYNTHETIC_mgp-tcn-datadump_labs_vitals_covs_na_thres_500_min_length_None_max_length_None_horizon_0_split_0.pkl", "wb") as fh:
        pickle.dump(mgp, fh)
    return out_dir


def main() -> None:
    path = generate()
    print(f"Wrote synthetic MIMIC-shaped files to {path}")


if __name__ == "__main__":
    main()
