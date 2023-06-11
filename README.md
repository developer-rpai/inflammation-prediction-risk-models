# Early prediction of inflammation using gradient boosting and deep learning

Inflammation is a fundamental biological response to harmful stimuli, but when dysregulated, it can lead to severe systemic issues and organ damage. Its management is highly time-sensitive because delayed treatment can increase morbidity and healthcare costs due to escalating systemic damage.

This project aims to analyze inflammation-related ICU data and predict its onset using machine learning, framing the detection as a supervised classification task. It uses time series data containing laboratory and vital parameters from patients' ICU stays.

To determine inflammation labels, we use a modified physiological criteria on an hourly basis, which requires evidence of a systemic response. These events occur when:

* **Suspicion of Inflammatory Response**:
    * If a relevant laboratory sample was obtained before a clinical intervention, then the treatment had to be ordered within 72 hours.
    * If the treatment was administered first, the sampling had to follow within 24 hours.
* **Systemic Dysfunction**: When the SOFA score shows an increase of at least 2 points, indicating an acute inflammatory impact on organ systems.

## Data

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

1.  **Install dependencies.** Install by running `pip install -r requirements.txt`. Python version used is 3.8.10.

2.  **Install PostgreSQL locally.** See the [PostgreSQL downloads](https://www.postgresql.org/download/) page for your system. PostgreSQL v12.12 and Ubuntu 20.04.4 were used in the project.

3.  **Accessing and building MIMIC-III.** For more details see [MIT Getting Started documentation](https://mimic.mit.edu/docs/gettingstarted/). Steps are:
    1.  Become a credentialed user on PhysioNet.
    2.  Complete required training.
    3.  Sign the required data use agreement.
    4.  Download files from the MIMIC-III website.
    5.  Build MIMIC-III locally.

4.  **Clone this repository.** To clone it from the command line, run:
  ```git clone https://github.com/developer-rpai/inflammation-prediction-risk-models.git```

5.  **Run experiments.** Requires creating a folder named `input` in the base project folder. Available experiments are:

| Model type | Experiment | Command |
| :--- | :--- | :--- |
| Gradient boosting (XGBoost) | Tuning | `python3 ./src/experiments/xgboost_experiments.py tuning` |
| Gradient boosting (XGBoost) | Test | `python3 ./src/experiments/xgboost_experiments.py test` |
| Recurrent neural networks | Tuning | `python3 ./src/experiments/rnn_experiments.py tuning` |
| Recurrent neural networks | Test | `python3 ./src/experiments/rnn_experiments.py test` |

---

## Project structure

	├── configs                               <- Configuration files for the experiments.
	├── input                                 <- Data files used by the models.
	│   ├── rnn
	│   │   ├── test
	│   │   ├── train
	│   │   └── val
	│   │
	│   └── xgboost
	│   	├── test
	│   	├── train
	│   	└── val
	│
	├── logs                                  <- TensorBoard logs.
	│   ├── hyperparam_opt                    <- Hyperparameter tuning logs.
	│   └── train                             <- Training logs.
	│
	├── models                                <- Optimal hyperparameters for the models.
	│
	├── notebooks                             <- Jupyter notebooks with EDA.
	│   ├── files_preview.ipynb
	│   ├── static_variables_eda.ipynb
	│   └── time_series_eda.ipynb
	│
	├── output                                <- Output files from the tuning, training and evaluation.
	│   ├── rnn
	│   └── xgboost
	│
	├── src                                   <- Source code for use in this project.
	│   ├── __init__.py                       <- Makes src a Python module.
	│   │
	│   ├── experiments                       <- Scripts to run the performed experiments.
	│   │   ├── rnn_experiments.py
	│   │   └── xgboost_experiments.py
	│   │
	│   ├── models                            <- Models' implementation.
	│   │
	│   ├── preprocessing                     <- Preprocessing scripts for EDA, training and evaluation.
	│   │   ├── __init__.py                   <- Makes preprocessing a Python module.
	│   │   ├── bin_and_impute.py             <- Binning and imputation of time series' missing values.
	│   │   ├── collect_records.py            <- ICU stays and patients' collection from different data files.
	│   │   ├── main_preprocessing.py         <- Data generation and loading.
	│   │   ├── rnn_preprocessing.py          <- Data padding, loading and label creation.
	│   │   ├── xgboost_preprocessing.py      <- Variables transformation into their stats, loading and label creation.
	│   │   └── util.py
	│   │
	│   ├── visualization                     <- Scripts to create exploratory and results oriented visualizations.
	│   │   ├── __init__.py                   <- Makes visualization a Python module.
	│   │   ├── plots.py                      <- Plots functions used in the project.
	│   │   └── util.py 
	│   │
	│   ├── train_rnn.py
	│   └── train_xgboost.py
	│
	└── requirements.txt                      <- Packages required to reproduce the project's working environment.

## Acknowledgements

* MIMIC-III and PhysioNet publications and website.
* [MIT-LCP code for building MIMIC](https://github.com/MIT-LCP/mimic-code).