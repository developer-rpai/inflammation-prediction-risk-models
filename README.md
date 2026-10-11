# Early prediction of inflammation using gradient boosting and deep learning

**Predicts the onset of inflammation 7 hours early from ICU vital-sign and lab
time series** (44 clinical variables, SOFA-based onset labels). Headline
benchmarks on 600 synthetic patients: **XGBoost test AUROC 0.997 · GRU test
AUROC 0.993**. The full pipeline runs end-to-end on synthetic data — no
PhysioNet credential, no PHI, no data download.

Inflammation is a fundamental biological response to harmful stimuli, but when dysregulated, it can lead to severe systemic issues and organ damage. Its management is highly time-sensitive because delayed treatment can increase morbidity and healthcare costs due to escalating systemic damage.

This project aims to analyze inflammation-related ICU data and predict its onset using machine learning, framing the detection as a supervised classification task. It uses time series data containing laboratory and vital parameters from patients' ICU stays.

To determine inflammation labels, we use a modified physiological criteria on an hourly basis, which requires evidence of a systemic response. These events occur when:

* **Suspicion of Inflammatory Response**:
    * If a relevant laboratory sample was obtained before a clinical intervention, then the treatment had to be ordered within 72 hours.
    * If the treatment was administered first, the sampling had to follow within 24 hours.
* **Systemic Dysfunction**: When the SOFA score shows an increase of at least 2 points, indicating an acute inflammatory impact on organ systems.

## Quickstart (synthetic data, no credentials needed)

The full pipeline now runs end-to-end on **synthetic ICU data** -- no
PhysioNet credential, no PostgreSQL, no data download. The generator produces
hourly vital-sign and lab time series for synthetic ICU stays (44 clinical
variables: 15 vitals + 29 labs) and labels inflammation onset with the same
physiological logic described above (culture/antibiotic suspicion window +
SOFA increase >= 2). The task is early prediction with a 7-hour horizon:
given the 48 hours before a prediction time, flag inflammation 7 hours
before onset.

```bash
pip install inflammation-prediction-risk-models
inflammation-train            # generate data, train XGBoost, print metrics
inflammation-train --model gru  # recurrent GRU on raw 48h sequences (NumPy only)
inflammation-train --n-patients 1000 --seed 7   # bigger cohort
pytest tests/                  # run the test suite (from a source checkout)
```

From source:

```bash
pip install -r requirements.txt
python -m inflammation_risk_models.train
```

`inflammation-train` accepts `--data-dir` pointing at a `cohort.parquet`/`cohort.csv`
in the same long-format schema the generator produces
(`patient_id, hour, <44 variables>, culture_sampled, antibiotic_given, sofa,
onset_hour`), so the same code trains on real extracted data when available.
Model, metrics, and feature importances are written to `output/synthetic/`.

## Benchmarks (synthetic data)

Reference run: 600 synthetic patients, seed 42, 7-hour prediction horizon.
XGBoost trains on per-variable summary statistics; the GRU trains directly
on the raw 48-hour hourly sequences (forward-filled, median-imputed,
z-scored with training-set statistics).

| Model | Test acc | Test AUROC | Test AUPRC |
| :--- | :--- | :--- | :--- |
| XGBoost (gradient boosting) | 0.989 | 0.997 | 0.983 |
| GRU (NumPy, 32 hidden units) | 0.978 | 0.993 | 0.961 |

The GRU is implemented from scratch in `inflammation_risk_models/gru.py` (single-layer GRU +
sigmoid head, mini-batch Adam, BPTT) -- no deep-learning framework needed,
so the recurrent path runs anywhere the synthetic pipeline does.

## Data in this repository

Real MIMIC-III rows are not stored here. PhysioNet credentialing, the local
path a credentialed extract should use, and the seeded mock tables are
documented in [DATA.md](DATA.md). Exploratory notebooks read
`data/synthetic/SYNTHETIC_*` files produced by
`scripts/generate_synthetic_mimic.py`. Saved notebook outputs are stripped on
commit (see `.pre-commit-config.yaml` and `.gitattributes`).

This project uses [MIMIC-III v1.4](https://physionet.org/content/mimiciii/1.4/) database with extraction and preprocessing modified scripts from [Machine Learning and Computational Biology Lab](https://github.com/BorgwardtLab/mgp-tcn). Data files are not provided, as [MIT-LCP](https://lcp.mit.edu/) requires to preserve the patients' privacy. MIMIC-III includes over 58,000 hospital admissions of over 45,000 patients, as encountered between June 2001 and October 2012.

### Extraction and filtering

Patients that fulfill any of these conditions are excluded from the final data set: under the age of 15, no chart data available, or logged via CareVue. To ensure that controls are not inflammation cases that developed the condition shortly before ICU, they are required not to be labeled with any relevant ICD-9 billing codes.

Cases that develop inflammation earlier than seven hours into their ICU stay are excluded as we aim for an early prediction of the condition. This enables a prediction horizon of 7h.

The final data set contains 570 inflammation cases and 5618 control cases.

### Missing values imputation

Missing values in clinical data is a constant problem that also appears in inflammation prediction. We apply different imputation approaches based on the models trained:

* **Gradient boosting models**: As time series data cannot be fed directly, we apply a time series encoding scheme that transforms each variable into a set of statistics representing its distribution: count, mean, std, min, max, and quantiles.
* **Recurrent neural networks**: We impute missing data using forward filling and each variable's median. They also require fixed-length data, so we apply padding to each time series with a masking value ignored during training. 

## Models

We create multiple gradient boosting and recurrent neural network models (using LSTM and GRU). For each type of model, we create baselines and tuned versions. We also create the following specific models:

* Gradient boosting with hyperparameters tuned based on the different previous hours of the prediction horizon before the inflammation onset.
* Stacked dense layers and stacked recurrent layers.

We use 44 clinical variables: 15 vital parameters and 29 laboratory parameters. For XGBoost models, we obtain 309 variables after encoding (including each time series length).

---

## Usage

1.  **Install.** Python >= 3.9. From a checkout, `pip install -r requirements.txt`, or install the package with `pip install inflammation-prediction-risk-models`.

2.  **Train on synthetic data.** The installed command and the module entry point accept the same flags (`--model {xgboost,gru}`, `--n-patients`, `--seed`, `--data-dir`, `--out-dir`, plus the `--gru-*` and `--n-estimators` options):

    ```bash
    inflammation-train
    python -m inflammation_risk_models.train --model gru --n-patients 1000 --seed 7
    ```

3.  **Optional: a credentialed MIMIC-III extract.** Real MIMIC-III is not in this repository. PostgreSQL and the PhysioNet download are only needed if you build your own extract. See [DATA.md](DATA.md) and the [MIT-LCP getting started guide](https://mimic.mit.edu/docs/gettingstarted/).

4.  **Clone.** `git clone https://github.com/developer-rpai/inflammation-prediction-risk-models.git`

The legacy experiment modules do not dispatch on a `tuning` or `test` argument. Their `__main__` blocks leave the original experiment calls commented out, and those calls expect a local credentialed extract (see [DATA.md](DATA.md)):

| Model type | Module |
| :--- | :--- |
| Gradient boosting (XGBoost) | `python -m inflammation_risk_models.experiments.xgboost_experiments` |
| Recurrent neural networks | `python -m inflammation_risk_models.experiments.rnn_experiments` |

---

## Project structure

```text
├── configs/                              <- Experiment configuration files.
├── data/synthetic/                       <- Seeded SYNTHETIC_* mock tables (see DATA.md).
├── inflammation_risk_models/             <- Installable package.
│   ├── synthetic_data.py                 <- Synthetic ICU cohort generator.
│   ├── preprocessing.py                  <- Time-series statistics encoding and splits.
│   ├── sequences.py                      <- Raw 48h sequence builder for the GRU path.
│   ├── gru.py                            <- NumPy GRU classifier.
│   ├── train.py                          <- CLI (`inflammation-train` / `python -m inflammation_risk_models.train`).
│   ├── experiments/
│   │   ├── rnn_experiments.py
│   │   └── xgboost_experiments.py
│   └── visualization/
│       ├── plots.py
│       └── util.py
├── models/hyperparam_opt/xgboost/        <- Saved hyperparameter result tables.
├── notebooks/
│   ├── files_preview.ipynb
│   ├── load_data_eda.ipynb
│   ├── static_variables_eda.ipynb
│   ├── time_series_eda.ipynb
│   └── rnn.ipynb
├── output/
│   ├── rnn/
│   └── xgboost/
├── scripts/generate_synthetic_mimic.py   <- Regenerates data/synthetic/.
├── tests/
│   ├── test_synthetic.py
│   ├── test_pipeline.py
│   ├── test_gru.py
│   └── test_mock_mimic.py
├── DATA.md
├── requirements.txt
└── pyproject.toml
```

## Related projects

* [SIDHA](https://github.com/developer-rpai/sidha) -- Sensor-to-Insight Data for
  Hidradenitis suppurativa Architecture: an open reference implementation for
  mapping episodic clinical records into continuous, research-ready
  inflammatory-disease intelligence (EHR ingestion via CSV/FHIR R4/OMOP CDM,
  computable clinical phenotypes, diagnostic-delay analytics, synthetic
  datasets). This repo's synthetic-data approach to inflammation modeling
  complements SIDHA's methods work.

## Acknowledgements

* MIMIC-III and PhysioNet publications and website.
* [MIT-LCP code for building MIMIC](https://github.com/MIT-LCP/mimic-code).