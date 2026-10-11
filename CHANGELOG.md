# Changelog

## 0.1.2 — 2026-10-11

- Exploratory notebooks run top to bottom on the seeded synthetic mock tables with current matplotlib, pandas, and seaborn. Saved outputs stay stripped.
- README, requirements comments, and the training CLI docstring match the installable package: Python >= 3.9, `inflammation_risk_models` instead of `src/`, and `inflammation-train` / `python -m inflammation_risk_models.train`.
- GitHub Actions runs pytest and a package build on Python 3.10, 3.11, and 3.12.

## 0.1.1

- Synthetic-data training pipeline (`inflammation-train`) and PyPI package metadata.
