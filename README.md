# credit visable

Lightweight project scaffold for credit risk analytics using the **Home Credit Default Risk** dataset.

This repository is intentionally structured for iterative work:

- develop locally in VS Code
- sync code and notebooks through GitHub
- run heavier experiments later in Google Colab

The current scaffold focuses on architecture only. It does **not** implement the full credit scoring pipeline yet.

## Project Objective

Build a clean, modular Python project that can grow into an end-to-end credit scoring workflow covering:

- data loading
- exploratory data analysis (EDA)
- preprocessing
- feature engineering
- WOE / IV / binning
- baseline modeling
- advanced modeling
- explainability
- fairness checks
- scorecard / cutoff analysis

## Workflow

1. Develop modules and notebooks locally in VS Code.
2. Push changes to GitHub for version control and collaboration.
3. Clone the same repository in Colab and continue analysis or heavier model training.

The package uses a `src` layout so imports stay predictable across local scripts, notebooks, and Colab.

## Folder Structure

```text
credit visable/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── configs/
│   └── base.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
│   ├── 00_colab_setup.ipynb
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling_baseline.ipynb
│   ├── 04_modeling_advanced.ipynb
│   ├── 05_xai_fairness.ipynb
│   └── 06_scorecard_cutoff.ipynb
├── reports/
│   └── figures/
├── src/
│   └── credit_visable/
│       ├── config.py
│       ├── data/
│       ├── explainability/
│       ├── features/
│       ├── governance/
│       ├── modeling/
│       ├── scoring/
│       └── utils/
└── tests/
```

## Setup

### Local environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Quick validation

```bash
pytest
```

## Data Placement

Project analysis code reads local CSV files from `data/raw/`.

Expected starter filenames are configured in [`configs/base.yaml`](configs/base.yaml).

Example:

```text
data/raw/application_train.csv
data/raw/application_test.csv
data/raw/bureau.csv
```

Place your uploaded or copied raw tables directly under `data/raw/` before
running notebooks or modules.

Common starter files:

```text
data/raw/application_train.csv
data/raw/application_test.csv
data/raw/bureau.csv
data/raw/bureau_balance.csv
data/raw/previous_application.csv
data/raw/installments_payments.csv
data/raw/credit_card_balance.csv
data/raw/POS_CASH_balance.csv
```

Additional helper files such as `sample_submission.csv` and
`HomeCredit_columns_description.csv` may also exist in `data/raw/`, but they
are not part of the default analysis input set.

## Running in Colab

1. Push this repository to GitHub.
2. Open Google Colab.
3. Clone the repository.
4. Install the package in editable mode.
5. Open notebooks from the cloned repo.

Typical Colab flow:

```python
!git clone https://github.com/<your-username>/credit-visable.git
%cd credit-visable
!pip install -U pip
!pip install -r requirements.txt
!pip install -e .
```

If you store the dataset in Google Drive, mount Drive and copy or point notebook code to the raw data directory before running analysis.

## Starter Modules

- `credit_visable.data`: loading tables and basic memory helpers
- `credit_visable.features`: preprocessing and IV / WOE placeholders
- `credit_visable.modeling`: baseline training, optional advanced-model training, evaluation
- `credit_visable.explainability`: SHAP integration placeholder
- `credit_visable.governance`: fairness summary placeholder
- `credit_visable.scoring`: scorecard / PDO placeholder
- `credit_visable.utils`: centralized path handling

## Development Notes

- Keep path handling centralized in `credit_visable.utils.paths`.
- Keep heavy business logic out of notebooks where possible.
- Use notebooks for exploration and modules for reusable logic.
- Add advanced dependencies only when needed.

## Next Development Steps

1. Add dataset-specific EDA in `notebooks/01_eda.ipynb`.
2. Build reusable preprocessing in `src/credit_visable/features/preprocess.py`.
3. Implement first-pass IV / WOE utilities for key variables.
4. Train a logistic regression baseline and track evaluation outputs.
5. Tune and extend tree-based modeling beyond the initial Phase 4 advanced baseline.
6. Extend explainability, fairness, and scorecard modules incrementally.

## Status

This repository now includes a starter workflow through Phase 4 advanced modeling. Later stages such as SHAP, fairness auditing, and score scaling still contain scaffold-heavy modules and `TODO` markers on purpose.
