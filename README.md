# credit visable

`credit visable` is a credit-risk modeling repository built on the Home Credit Default Risk dataset. The hardened path in this repo now targets an application-stage binary default model with a governed workflow:

- raw application data ingestion
- application-only feature engineering and data-quality repair
- protected-feature deny list for model training
- `development / calibration / holdout test` partitioning
- feature review with IV / correlation / VIF / PCA diagnostics
- calibrated XGBoost champion training
- frozen score scaling, risk bands, and cutoff policy
- SHAP explainability, fairness review, and monitoring baselines

The objective is not just to maximize ranking metrics. The objective is to produce a scoring workflow that is reproducible, auditable, and usable in underwriting discussion.

## Objective

- Business problem: application-stage credit default classification
- Target: `TARGET = 1` means the applicant showed payment difficulty in the first installment periods
- Current scope: `application_train.csv` only
- Deferred scope: bureau / behavioral monthly aggregation and true out-of-time validation

## Governed Pipeline

The production-oriented path is the governed application pipeline exposed from [`src/credit_visable/scoring/phase6_reporting.py`](/Users/kingjason/资源/credit%20visable/src/credit_visable/scoring/phase6_reporting.py).

```text
application_train.csv
-> application feature engineering
-> training feature policy (protected and restricted features removed from model inputs)
-> development / calibration / holdout test split
-> preprocessing + clipping + one-hot encoding
-> IV / correlation / VIF / PCA review artifacts
-> XGBoost tuning inside development only
-> Platt calibration on calibration split only
-> holdout evaluation
-> score scaling + frozen risk bands + cutoff policy
-> SHAP + fairness + monitoring artifacts
```

## Governance Rules

- Protected and sensitive features are retained for review only and are excluded from training mode feature selection.
- Holdout rows are not used for tuning, calibration, or risk-band construction.
- Risk-band thresholds are frozen from the development population and then applied unchanged to calibration, holdout, and scoring outputs.
- The repository does not claim genuine OOT validation unless a true application timestamp exists.
- Score cutoff economics are versioned, but current assumptions remain provisional until finance approval.

## Main Entry Points

- [`src/credit_visable/features/application_features.py`](/Users/kingjason/资源/credit%20visable/src/credit_visable/features/application_features.py): sentinel cleanup, missing flags, affordability ratios, and application-only feature engineering
- [`src/credit_visable/features/preprocess.py`](/Users/kingjason/资源/credit%20visable/src/credit_visable/features/preprocess.py): governed preprocessing artifacts with `development / calibration / test` splits
- [`src/credit_visable/features/review.py`](/Users/kingjason/资源/credit%20visable/src/credit_visable/features/review.py): IV, correlation, VIF, and PCA diagnostics for feature review
- [`src/credit_visable/modeling/train_tree_models.py`](/Users/kingjason/资源/credit%20visable/src/credit_visable/modeling/train_tree_models.py): governed XGBoost training with internal validation and CV on development only
- [`src/credit_visable/scoring/pdo_scorecard.py`](/Users/kingjason/资源/credit%20visable/src/credit_visable/scoring/pdo_scorecard.py): score scaling and frozen risk-band helpers
- [`src/credit_visable/scoring/phase6_reporting.py`](/Users/kingjason/资源/credit%20visable/src/credit_visable/scoring/phase6_reporting.py): end-to-end governed champion run
- [`src/credit_visable/governance/monitoring.py`](/Users/kingjason/资源/credit%20visable/src/credit_visable/governance/monitoring.py): PSI-style monitoring baseline and fairness drift diagnostics
- [`configs/scorecard.yaml`](/Users/kingjason/资源/credit%20visable/configs/scorecard.yaml): calibration, score scaling, risk-band, and cutoff policy defaults

## Project Layout

```text
credit visable/
├── Agent.md
├── README.md
├── configs/
│   ├── base.yaml
│   └── scorecard.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
├── reports/
│   └── figures/
├── src/
│   └── credit_visable/
└── tests/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Quick Validation

```bash
pytest -q
```

## Data Placement

Expected raw files are configured in [`configs/base.yaml`](/Users/kingjason/资源/credit%20visable/configs/base.yaml):

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

The governed refactor currently trains only on `application_train.csv`. History tables remain available for future bureau and behavioral aggregation work.

## Running The Governed Pipeline

Programmatic execution:

```python
from credit_visable.scoring import run_governed_application_pipeline

result = run_governed_application_pipeline(
    feature_set_name="traditional_plus_proxy",
    shap_sample_size=512,
)

print(result["output_dir"])
print(result["summary"])
```

Primary governed outputs are written under:

```text
data/processed/ds_audit_refactor/governed_xgboost_traditional_plus_proxy/
```

Key artifacts include:

- `metrics_comparison.csv`
- `holdout_scores.csv`
- `frozen_risk_band_table.csv`
- `holdout_risk_band_summary.csv`
- `holdout_calibration_summary.csv`
- `holdout_cutoff_sweep.csv`
- `holdout_fairness_metric_summary.csv`
- `holdout_policy_group_summary.csv`
- `monitoring_summary.csv`
- `monitoring_fairness_drift.csv`
- `shap_raw_feature_contributions.csv`
- `shap_proxy_family_contributions.csv`
- `shap_local_case_explanations.csv`
- `adverse_action_reason_codes.csv`
- `run_manifest.json`
- `summary.json`

## Notebook Workflow

The notebook sequence is still available for exploratory and reporting use:

1. `00_colab_setup.ipynb`
2. `01_eda.ipynb`
3. `02_preprocessing.ipynb`
4. `03_modeling_baseline.ipynb`
5. `04_modeling_advanced.ipynb`
6. `05_xai_fairness.ipynb`
7. `06_scorecard_cutoff.ipynb`

Notebook outputs should now be interpreted against the governed pipeline rules above. The legacy notebook path remains useful for analysis, but the regulator-facing champion path is the governed application pipeline.

## What Changed In The DS Audit Refactor

- Added application-only feature engineering for known data issues such as `DAYS_EMPLOYED == 365243`, missingness flags, and affordability ratios.
- Introduced training-mode feature selection that removes protected and restricted proxy features from model inputs.
- Replaced the single validation split with `development / calibration / holdout test`.
- Added dedicated calibration on the calibration split and final evaluation on frozen holdout only.
- Added feature review artifacts for IV, correlation, VIF, and PCA diagnostics.
- Reworked scorecard logic to freeze risk-band thresholds on development before applying them to holdout and downstream scoring.
- Added SHAP reason-code style outputs, fairness review on final calibrated policy decisions, and monitoring baselines.
- Added reproducibility manifests with dataset fingerprint, split hashes, git commit, and model metadata.

## Limitations

- No genuine OOT validation is possible with the current raw data because a true application timestamp is not available.
- Multi-table bureau and behavioral aggregation is deferred; current champion scope is application-only.
- Economics assumptions in `scorecard.yaml` should be treated as provisional until business owners approve them.
- Fairness outputs are governance diagnostics, not a final legal or compliance certification.

## Colab

The repository can still be run in Colab, but the preferred dependency installation is now:

```python
!pip install -U pip
!pip install -r requirements.txt
!pip install -e .
```

Do not claim OOT or production readiness from notebook visuals alone. Use the governed artifacts under `data/processed/ds_audit_refactor/` as the auditable reference set.
