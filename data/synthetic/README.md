# Synthetic MIMIC-shaped tables

**These files are synthetic.** They were produced by
`scripts/generate_synthetic_mimic.py` (seed 42). Every value is drawn from a
NumPy random generator. They are not copies or perturbations of MIMIC-III
rows, and they are not a credentialed PhysioNet extract.

Identifiers are obviously fake:

| Field | Range |
| :--- | :--- |
| `subject_id` | starts at 900000 |
| `hadm_id` | starts at 910000 |
| `icustay_id` | starts at 920000 |

`920000` is the example ICU stay used by `notebooks/time_series_eda.ipynb`.

The three `SYNTHETIC_tvt_info_split_*.pkl` files are dicts with `train`,
`validation`, and `test` int64 arrays (lengths 4965, 620, and 620), the same
shape as the historical split files, filled from the 920000–926204 id pool.

Column layouts match the tables the exploratory notebooks load (static
demographics, hourly labs and vitals, SOFA, and the wide and short
`full_labvitals` modeling tables). The cohort is a small mock (on the order
of 100 stays), not a size-matched MIMIC extract. Regenerate with:

```bash
python scripts/generate_synthetic_mimic.py
```

Real MIMIC-III data is not stored in this repository. See `DATA.md`.
