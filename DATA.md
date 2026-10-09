# Data

Real MIMIC-III patient data is not stored in this repository. The notebooks,
split files, and training pipeline in the current tree use synthetic data only.

## MIMIC-III requires PhysioNet credentialing

[MIMIC-III v1.4](https://physionet.org/content/mimiciii/1.4/) is restricted. A
credentialed user has to:

1. Create a PhysioNet account and become a credentialed user.
2. Complete the required human-subjects training.
3. Sign the data use agreement.
4. Download MIMIC-III from PhysioNet.
5. Build a local database (PostgreSQL is what the original extraction used).
   The [MIT-LCP MIMIC getting started guide](https://mimic.mit.edu/docs/gettingstarted/)
   describes that process.

Do not commit the download, database dumps, extracted stay tables, identifier
lists, notebook outputs, model artifacts, or TensorBoard logs from a real
extract. This repository's history is not rewritten by the change that removed
those files from the current tree; a separate history purge is an owner decision.

## What this repository contains instead

`scripts/generate_synthetic_mimic.py` writes mock tables under
`data/synthetic/`. Every filename starts with `SYNTHETIC_`, and
`data/synthetic/README.md` describes the schema. Identifiers start at 900000.
The exploratory notebooks read those files.

The installable package (`inflammation-train`) does not read those tables. It
generates its own hourly cohort in `inflammation_risk_models.synthetic_data`
and writes metrics under `output/synthetic/` (gitignored).

## Plugging in a credentialed extract

The legacy experiment scripts still expect a local extract you produce
yourself, after credentialing. They are not pointed at the synthetic mocks,
so a paper-style run cannot silently train on the mock cohort.

| Script | Path it reads |
| :--- | :--- |
| `inflammation_risk_models/experiments/xgboost_experiments.py` | `output/full_labvitals_horizon_0_dropped_short.csv` |
| RNN and XGBoost experiment entry points | train/validation/test arrays under an `input/` directory you create |
| Notebooks, if you are not using the mocks | the same column layouts as `data/synthetic/SYNTHETIC_*.csv` |

Replace only your local copies. Keep real identifier lists and trained
artifacts out of git. Notebook outputs are stripped by the `nbstripout`
pre-commit hook (`.pre-commit-config.yaml`) and by the git filter in
`.gitattributes`.

Install the hook once per clone:

```bash
pip install nbstripout pre-commit
nbstripout --install
pre-commit install
```

`nbstripout --install` registers the filter named in `.gitattributes`, so a
later `git add` of a notebook drops saved outputs before they are committed.
