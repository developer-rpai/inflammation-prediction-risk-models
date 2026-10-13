"""Guards for the synthetic MIMIC-shaped tables.

The committed mocks must stay obviously fake: identifiers at or above 900000,
never in the MIMIC-III icustay range, and reproducible from the seeded script.
"""

import importlib.util
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SYN = ROOT / "data" / "synthetic"
MIMIC_ICUSTAY = re.compile(r"\b2\d{5}\b")


def _generator():
    spec = importlib.util.spec_from_file_location(
        "generate_synthetic_mimic", ROOT / "scripts" / "generate_synthetic_mimic.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_split_pickles_use_fake_ids():
    for i in range(3):
        with open(SYN / f"SYNTHETIC_tvt_info_split_{i}.pkl", "rb") as fh:
            split = pickle.load(fh)
        assert set(split) == {"train", "validation", "test"}
        assert split["train"].shape == (4965,)
        assert split["validation"].shape == (620,)
        assert split["test"].shape == (620,)
        for arr in split.values():
            assert arr.dtype == np.int64
            assert arr.min() >= 920_000
            assert arr.max() < 930_000
            assert not ((arr >= 200_000) & (arr < 300_000)).any()


def test_static_tables_schema_and_fake_ids():
    static = pd.read_csv(SYN / "SYNTHETIC_static_variables.csv")
    cases = pd.read_csv(SYN / "SYNTHETIC_static_variables_cases.csv")
    controls = pd.read_csv(SYN / "SYNTHETIC_static_variables_controls.csv")
    assert list(static.columns)[:3] == ["subject_id", "hadm_id", "icustay_id"]
    assert "icustay_id.1" in cases.columns
    assert "matched_case_icustay_id" in controls.columns
    for frame in (static, cases, controls):
        assert frame["subject_id"].min() >= 900_000
        assert frame["hadm_id"].min() >= 910_000
        assert frame["icustay_id"].min() >= 920_000
    short = pd.read_csv(SYN / "SYNTHETIC_full_labvitals_horizon_0_dropped_short.csv")
    assert short.shape[1] == 49
    assert (short["icustay_id"] == 920_000).any()
    assert "SYNTHETIC" in (SYN / "README.md").read_text()


def test_generator_is_seeded_and_not_a_copy_of_real_ranges():
    gen = _generator()
    stays_a = gen.build_stays(np.random.default_rng(gen.SEED))
    stays_b = gen.build_stays(np.random.default_rng(gen.SEED))
    pd.testing.assert_frame_equal(stays_a, stays_b)
    assert stays_a["icustay_id"].min() == gen.ICUSTAY_ID_START
    assert int(stays_a.iloc[0]["icustay_id"]) == gen.EXAMPLE_ICUSTAY_ID


def test_notebooks_have_no_saved_outputs_or_mimic_icustay_ids():
    for path in (ROOT / "notebooks").glob("*.ipynb"):
        text = path.read_text()
        nb = json.loads(text)
        for cell in nb["cells"]:
            assert not cell.get("outputs"), path.name
        assert MIMIC_ICUSTAY.search(text) is None, path.name
