# credit visable

Notebook-driven credit risk analytics workflow for the **Home Credit Default Risk** dataset.

This repository is intentionally structured for a Colab-first delivery loop:

- develop and validate locally
- push notebook and module updates to GitHub `main`
- clone the same repository in Google Colab as the canonical execution environment

## Project Objective

Build a clean, modular Python project that supports an end-to-end credit scoring workflow covering:

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

1. Develop modules and notebooks locally.
2. Push the current state to GitHub `main`.
3. Clone the same repository in Colab and run the phase notebooks there.

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

1. Push the latest local state to `origin/main`.
2. Open Google Colab.
3. Clone the repository from GitHub.
4. Install the package plus the Colab runtime extras.
5. Mount Google Drive if your raw CSVs live there.
6. Open `notebooks/00_colab_setup.ipynb` and set `RAW_DATA_DIR` by hand for that run.

Typical Colab flow:

```python
!git clone https://github.com/GDSherlock/Make-Credit-Visable.git
%cd Make-Credit-Visable
!pip install -U pip
!pip install -r requirements.txt
!pip install -e .
!pip install xgboost shap
```

If you store the dataset in Google Drive, mount Drive and point the notebook-local `RAW_DATA_DIR` to the actual folder, for example:

```python
from pathlib import Path
RAW_DATA_DIR = Path("/content/drive/MyDrive/home-credit/data/raw")
```

`RAW_DATA_DIR` is a run-time input. Do not commit per-session Drive paths back into the repo.

## Phase Outputs

The notebook workflow writes standard artifacts under:

- `data/processed/phase0_environment/`
- `data/processed/eda/`
- `data/processed/preprocessing/`
- `data/processed/modeling_baseline/`
- `data/processed/modeling_advanced/`
- `data/processed/xai_fairness/`
- `data/processed/scorecard_cutoff/`
- `reports/figures/`

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

## Status

This repository includes a notebook workflow through Phase 6 score / cutoff analysis. The intended path is:

- `Phase 0`: environment and dataset validation
- `Phase 1`: EDA with standard artifact export
- `Phase 2`: dual feature-set preprocessing
- `Phase 3`: logistic baseline comparison
- `Phase 4`: XGBoost-based advanced modeling comparison
- `Phase 5`: explainability plus grouped fairness / governance review
- `Phase 6`: notebook-local score transform, risk bands, and cutoff analysis

Some repository modules are still intentionally lightweight:

- multi-table customer-level aggregation is not the main pipeline yet
- `credit_visable.scoring.pdo_scorecard` remains placeholder-only
- fairness outputs are grouped governance summaries, not a full policy audit
