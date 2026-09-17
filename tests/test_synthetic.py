"""Smoke tests for the synthetic ICU cohort generator."""

import numpy as np
import pandas as pd

from src.synthetic_data import FEATURE_VARS, generate_cohort


def test_schema():
    df = generate_cohort(n_patients=20, seed=1)
    expected = (["patient_id", "hour"] + FEATURE_VARS
                + ["culture_sampled", "antibiotic_given", "sofa", "onset_hour",
                   "true_case"])
    assert list(df.columns) == expected
    assert df["patient_id"].nunique() == 20
    assert (df["sofa"] >= 0).all() and (df["sofa"] <= 24).all()
    assert set(df["culture_sampled"].unique()) <= {0, 1}


def test_both_classes_present():
    df = generate_cohort(n_patients=300, seed=42)
    onset = df.drop_duplicates("patient_id")["onset_hour"]
    case_rate = onset.notna().mean()
    assert 0.03 < case_rate < 0.30, f"case_rate={case_rate}"


def test_reproducible_with_seed():
    a = generate_cohort(n_patients=25, seed=123)
    b = generate_cohort(n_patients=25, seed=123)
    pd.testing.assert_frame_equal(a, b)


def test_onset_after_observation_window():
    df = generate_cohort(n_patients=100, seed=7)
    onsets = df.drop_duplicates("patient_id")["onset_hour"].dropna()
    assert (onsets >= 48).all(), "early-prediction framing needs onset >= 48h"


def test_cases_deteriorate():
    """Cases should show worse physiology near onset than at admission."""
    df = generate_cohort(n_patients=200, seed=11)
    first = df[df["hour"] == 0].set_index("patient_id")
    cases = df[(df["true_case"] == 1) & df["onset_hour"].notna()
               ].drop_duplicates("patient_id")
    pid = int(cases["patient_id"].iloc[0])
    onset = int(cases["onset_hour"].iloc[0])
    pat = df[df["patient_id"] == pid].set_index("hour")
    assert pat.loc[onset, "sofa"] > first.loc[pid, "sofa"]
    lac_near = pat["lactate"].loc[max(0, onset - 6):onset].dropna()
    lac_base = pat["lactate"].loc[0:12].dropna()
    if len(lac_near) and len(lac_base):
        assert lac_near.mean() > lac_base.mean()
