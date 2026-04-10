# credit visable

面向 **Home Credit Default Risk** 数据集的 notebook-driven credit risk workflow。这个仓库的重点不是堆砌模型，而是把 `EDA`、`preprocessing`、`baseline`、`advanced modeling`、`XAI`、`fairness / governance`、`score / cutoff` 串成一条可复核、可解释、可继续在 Colab 跑通的分析链路。

仓库默认采用本地开发 + Colab 复现的交付方式：

- 本地开发和验证 notebook / module
- 推送到 GitHub `main`
- 在 Google Colab 中复现同一套 phase workflow

## Project Objective

项目目标是构建一套围绕信用评分场景的分析流程，既能输出模型结果，也能把结果翻译成业务语言，例如 `PD`、`score`、`risk band`、`approval rate`、`cutoff impact`。

## Phase Guide（0-6）

### Phase 0: Environment and Dataset Validation

- 做了什么：确认运行环境是本地还是 Colab，检查 Python / package import、原始数据目录、以及 `configs/base.yaml` 里配置的核心表是否存在。
- 用了什么方法：逐个执行依赖导入检查，扫描 `data/raw/` 下的预期表，记录运行时路径、模块可用性和 explainability / tree backend 就绪状态。
- 为什么看这些 chart：这一阶段默认不需要图；用 `table availability` 和 `import status` 表格就足够，因为它的作用是 phase gate，而不是业务分析。
- 产出什么：`data/processed/phase0_environment/runtime_summary.json`、`table_availability.csv`、`import_status.csv`。
- 结论怎么读：这里的结论属于 `descriptive finding`，只回答“环境和数据是否已满足进入正式分析的前置条件”，不回答模型效果问题。
- 报告怎么用：先看 `runtime_summary.json` 确认 `raw_data_dir`、运行环境和依赖是否齐全；再看 `table_availability.csv` 判断 `application_train.csv` 以及历史表是否可用；确认无阻塞后再进入 `Phase 1`。

### Phase 1: Exploratory Data Analysis

- 做了什么：先对 `application_train.csv` 做主表深度 `EDA`，再对 `bureau`、`bureau_balance`、`previous_application`、`installments_payments`、`credit_card_balance`、`POS_CASH_balance` 做 overview；当前不做客户级聚合建模。
- 用了什么方法：在主表中构造分析级派生字段 `AGE_YEARS`、`DAYS_EMPLOYED_CLEAN`、`YEARS_EMPLOYED`、`INCOME_CREDIT_RATIO`、`ANNUITY_INCOME_RATIO`；其中 `DAYS_EMPLOYED == 365243` 会先视为哨兵值并转成缺失后再生成 `YEARS_EMPLOYED`；同时输出 descriptive stats、missingness、numeric summary、grouped default-rate slice 和历史表概览。
- Method Notes：这些字段属于 `EDA-derived analytical fields`，用于解释和诊断，不是当前 `Phase 2` 建模输入的正式特征；当前建模主流程仍以 `application_train` 原始列为基础。
- 为什么看这些 chart：`TARGET distribution` 用来判断样本不平衡，因此后续 metric 要重看 `PR` / `KS` 而不是只看 `accuracy`；`Top missingness` 用来识别高缺失字段和 `Phase 2` 的插补压力；`numeric distributions` 用来看收入、授信、年金、年龄等变量的偏态、异常值和量纲差异；`boxplots by TARGET` 用来看好坏样本在关键数值变量上的分离度；`correlation heatmap` 用来看 `AMT_*`、比率变量和 `EXT_SOURCE_*` 之间的相关性与冗余；`category business view` 用来看 `CODE_GENDER`、`NAME_FAMILY_STATUS`、`NAME_EDUCATION_TYPE`、`NAME_INCOME_TYPE`、`OCCUPATION_TYPE`、`ORGANIZATION_TYPE`、`FLAG_OWN_CAR`、`FLAG_OWN_REALTY` 的样本结构与坏账率；`age band` 和 `annuity-income band` 图把风险解释转成 `life-cycle` / `affordability` 语言；历史表图则用于判断后续是否值得进入多表聚合，而不是现在就直接建模。
- 历史表图的具体目的：`bureau` 看历史授信活跃状态和客户历史负债密度；`bureau_balance` 看月度状态分布和逾期轨迹密度；`previous_application` 看过往申请类型、申请状态与申请额/授信额关系；`installments_payments` 看逾期次数和还款纪律；`credit_card_balance` 与 `POS_CASH_balance` 看账户使用强度和记录密度。
- 产出什么：`data/processed/eda/eda_summary.json`、`top_missingness.csv`、`numeric_distribution_summary.csv`、`numeric_by_target_median.csv`、`correlation_matrix.csv`、`fairness_summary.csv`、`iv_summary.csv`、`age_woe_detail.csv`、`age_income_default_heatmap.csv`、`income_family_default_heatmap.csv`、`history_table_overview.csv`，以及 `reports/figures/phase1_*.png`。
- 结论怎么读：这里的结论仍然是 `descriptive finding`，重点是变量质量、缺失、风险切片和 proxy-sensitive 信号，不应把 EDA 观察直接表述为 production policy。
- 报告怎么用：先看 `eda_summary.json` 和 `checkpoint.csv` 抓整体状态；再看 `top_missingness.csv`、`correlation_matrix.csv`、`fairness_summary.csv` 等表确认变量质量与风险切片；最后看 `reports/figures/phase1_*.png` 做业务解释，并把“高缺失字段、相关性强字段、proxy-sensitive 字段”带入 `Phase 2`。

### Phase 2: Preprocessing

- 做了什么：围绕 `application_train` 主表构建两套固定 `data regime`，并生成可复用的 train / valid preprocessing artifact，供 `Phase 3` 和 `Phase 4` 直接复用。
- 用了什么方法：`traditional_core` 去掉 `TARGET`、`SK_ID_CURR` 和 proxy families；`traditional_plus_proxy` 保留全部主表特征；proxy families 当前包括 `EXT_SOURCE_*`、`OBS/DEF_*_CNT_SOCIAL_CIRCLE`、`FLAG_PHONE`、`FLAG_WORK_PHONE`、`FLAG_EMP_PHONE`、`FLAG_EMAIL`、`DAYS_LAST_PHONE_CHANGE`、`ORGANIZATION_TYPE`、`OCCUPATION_TYPE`、`FLAG_DOCUMENT_*`，以及所有包含 `REGION` 或 `_CITY_` 的列。数值特征按 dtype 自动识别，类别特征按 `object` / `category` / `bool` 识别；`TARGET` 按分层方式做 `80/20` train / valid split；数值缺失值使用 `median imputation`；类别缺失值使用 `most_frequent imputation`；类别变量使用 `OneHotEncoder`；稀有类别按 `min_frequency=0.01` 合并到 `infrequent`；最终输出为 sparse matrix。
- Method Notes：当前 pipeline 的 missing value 处理是基础插补，不是高级插补；当前没有做 `missing indicator`、没有做 `scaling`、没有做 `quantile clipping`、也没有把 `WOE / IV` 真正落到主流程里。`Phase 2` 直接基于 `application_train` 原始列，不包含 `Phase 1` 的分析级派生字段。
- 为什么看这些 chart / summary：`feature family summary` 用来解释每套 feature set 的 `numeric` / `categorical` 构成；`missingness summary` 用来说明哪些字段进入了插补流程；`matrix density` 用来说明 one-hot 之后的维度膨胀与稀疏性，帮助理解为什么输出保存成 `.npz` 而不是单个 dense CSV。
- 产出什么：`data/processed/preprocessing/traditional_core/`、`data/processed/preprocessing/traditional_plus_proxy/`、每套目录下的 `X_train.npz`、`X_valid.npz`、`train_meta.csv`、`valid_meta.csv`、`feature_names.csv`、`manifest.json`、`feature_set_manifest.json`、`preprocessing_decision_manifest.json`，以及总览文件 `feature_catalog.csv`、`preprocessing_decision_summary.csv`、`processing_methods_summary.json`、`processing_methods_summary_cn.md`。
- 结论怎么读：这里的结论属于“建模输入已标准化”的阶段结论。后续如果比较模型，必须在固定 `feature set` 下比较；如果比较 data uplift，必须在固定模型家族下比较。
- 报告怎么用：先看 `processing_methods_summary_cn.md` 或 `preprocessing_summary.csv` 理解两套口径；再看每套的 `feature_set_manifest.json` 和 `manifest.json` 确认字段归属、矩阵形状和稀疏度；最后再进入 `Phase 3` 或 `Phase 4`，并保持 train / valid 对齐。

### Phase 3: Baseline Modeling

- 做了什么：固定 `logistic` 家族，在同一 validation split 上分别训练 `traditional_core` 和 `traditional_plus_proxy` 两套 baseline，只比较数据增量，不比较模型家族。
- 用了什么方法：直接复用 `Phase 2` 的标准 artifact，先验证两套 feature set 的 `train_meta` / `valid_meta` 是否主键与目标完全对齐，再训练 logistic baseline，输出 `ROC-AUC`、`average precision`、`precision`、`recall`、`F1`、`KS` 以及 threshold scenario。
- 为什么看这些 chart：`ROC` 用来看排序能力；`PR` 用来看少数类识别能力；`KS` 用来看 cutoff separation potential；`threshold scenarios` 用来看不同阈值下的 `approve / reject / approved bad rate / rejected bad capture` 的业务影响。
- 产出什么：`data/processed/modeling_baseline/comparison_metrics.csv`、`data_uplift_summary.csv`、`threshold_scenarios.csv`、`operating_threshold_comparison.csv`、`validation_score_comparison.csv`、`calibration_curve_points.csv`、`gain_curve_points.csv`、`lift_curve_points.csv`、每套目录下的 `metrics.json` / `validation_scores.csv`，以及 `reports/figures/phase3_*.png`。
- 结论怎么读：这里的结论属于 `model comparison finding`，但它回答的是“在同一 logistic 家族下，加不加 proxy 是否有数据增量”，不回答“advanced model 是否更强”。
- 报告怎么用：先看 `comparison_metrics.csv` 和 `data_uplift_summary.csv` 判断 proxy 加入后是否带来实质增量；再看 `threshold_scenarios.csv` 和 `operating_threshold_comparison.csv` 理解业务阈值影响；最后看 `phase3` 的 `ROC` / `PR` / `KS` 图，并把结论带入 `Phase 4` 的四模型对照。

### Phase 4: Advanced Modeling

- 做了什么：在两套固定 `feature set` 上训练同一个 `advanced backend`，并与 `Phase 3` 的 logistic baseline 组成四模型矩阵；当前已提交结果使用 `xgboost` backend。
- 用了什么方法：对 `traditional_core` 和 `traditional_plus_proxy` 复用同一 backend、同一超参数块，分别与各自的 logistic baseline 做对照；同时拆出 `model_uplift_summary` 和 `data_uplift_summary`，避免把“模型升级”和“数据口径变化”混成一个结论。
- 为什么看这些 chart：`four-model ROC / PR / KS` 是为了分开看 `model uplift` 和 `data uplift`；`feature importance` 是为了看 advanced model 依赖的是传统信贷变量，还是更多 proxy-sensitive families。
- 产出什么：`data/processed/modeling_advanced/comparison_metrics.csv`、`model_uplift_summary.csv`、`data_uplift_summary.csv`、`threshold_scenarios.csv`、`validation_score_comparison.csv`、`summary.json`、`best_candidate_summary.json`、`calibration_curve_points.csv`、`gain_curve_points.csv`、`lift_curve_points.csv`，以及 `reports/figures/phase4_*.png` 和 feature importance 图。
- 结论怎么读：这里的结论同样属于 `model comparison finding`。要分成两个问题读：同一 data regime 下，`advanced model` 是否明显优于 `logistic`；同一模型家族下，proxy 是否带来额外 uplift。它还不是 fairness 或 policy 结论。
- 报告怎么用：先看 `summary.json`、`model_uplift_summary.csv` 和 `data_uplift_summary.csv` 确定候选模型与对照关系；再看 `comparison_metrics.csv` 和 `validation_score_comparison.csv`；最后看四模型 `ROC` / `PR` / `KS` 图和 feature importance 图，并把“候选模型 + proxy uplift 风险”带入 `Phase 5`。

### Phase 5: XAI + Fairness / Governance Review

- 做了什么：围绕 `Phase 4` 的候选模型做 explainability，并对 proxy uplift 进行 grouped fairness / governance review；当前候选框架来自 `traditional_plus_proxy + advanced backend`。
- 用了什么方法：在当前仓库产物中，候选模型与 comparator 已先被对齐，然后分别生成 `SHAP` global explanation、`SHAP waterfall`、`LIME` local explanation、proxy family contribution，以及按 `age_band`、`family_status_group`、`region_rating_group`、`organization_group`、`occupation_group` 等分组的 grouped outcome summary。
- 为什么看这些 chart：`SHAP beeswarm` 看全局影响方向和强度；`SHAP bar` 看 mean absolute contribution；`proxy family contribution` 看 uplift 是否主要来自 proxy-sensitive 家族；`SHAP waterfall + LIME` 看单笔样本是如何被判高风险或低风险；`group fairness summary` 和对应条形图用来看不同 group 的实际坏账率、平均预测 `PD`、`approval rate` 是否出现结构性差异。
- 产出什么：`data/processed/xai_fairness/summary.json`、`candidate_model_selection.json`、`group_fairness_summary.csv`、`fairness_metric_summary.csv`、`proxy_uplift_summary.csv`、`top_shap_interactions.csv`、`candidate_partial_dependence.csv`、`validation_review_frame.csv`，以及 `reports/figures/phase5_*.png`。
- 结论怎么读：这里的结论属于 `governance diagnostic`，不是完整 `fairness audit`。它回答的是“候选模型为什么有效、proxy uplift 主要从哪里来、哪些 group 的 outcome gap 需要进入后续 cutoff 讨论”，而不是“模型已经满足全部治理要求”。
- 报告怎么用：先看 `candidate_model_selection.json` 和 `summary.json` 确认候选模型、matched comparator 和 explainability method；再看 `group_fairness_summary.csv`、`proxy_uplift_summary.csv`、`validation_review_frame.csv`；最后看 `phase5` 的 `SHAP`、`LIME` 和 grouped chart，并把 proxy-sensitive uplift 与 grouped gap 一起带入 `Phase 6`。

### Phase 6: Scorecard and Cutoff Analysis

- 做了什么：把 `Phase 5` 的候选模型输出翻译成 `score`、`risk band`、`cutoff`、`policy scenario` 语言，并把 candidate 与 comparator 的差异转成 band / decision migration。
- 用了什么方法：当前采用 `notebook-local score transform`，直接把候选模型的 predicted `PD` 映射成分数；默认参数为 `Base Score = 600`、`PDO = 20`、`Base Odds = 50:1`；同时生成 `A-E risk band`、score decile、`10-bin calibration summary`、`Brier score`、按 10 分步长构造的 `score cutoff grid`、`conservative / balanced / growth` 三段式 `policy scenario`，以及 `candidate vs comparator` 的 score / band / decision migration。
- 为什么看这些 chart：`score histogram + KDE` 和 `ECDF` 用来看分数分布是否过度集中；`PD-to-score curve` 用来解释分数与 `PD` 的单调映射；`calibration curve` 和 `decile reliability` 用来看预测 `PD` 与真实坏账率之间的偏差；`risk band count / default rate / mean PD` 用来把模型输出翻译成 policy language；`cutoff sweep` 三张图分别回答阈值变化如何影响 `approval rate`、`approved bad rate`、`rejected bad capture`；`policy scenario composition` 用来看 `conservative` / `balanced` / `growth` 三种策略的结构差异；`migration heatmap` 用来看从 comparator 到 candidate 的客户迁移；grouped `balanced policy` 图用来看策略层面的 governance sensitivity。
- 产出什么：`data/processed/scorecard_cutoff/xgboost_traditional_plus_proxy/summary.json`、`score_transform_meta.json`、`score_frame.csv`、`score_decile_summary.csv`、`risk_band_summary.csv`、`calibration_summary.csv`、`cutoff_sweep.csv`、`profit_curve.csv`、`profit_assumptions.json`、`optimal_profit_cutoff.json`、`optimal_profit_fairness_summary.csv`、`policy_scenarios.csv`、`policy_group_summary.csv`、`score_migration_matrix.csv`、`decision_migration_matrix.csv`，以及 `reports/figures/phase6_*.png`。
- 结论怎么读：这里的结论属于 `policy diagnostic finding`。它能帮助讨论分数、band、cutoff 和 grouped outcome，但不应被描述成 production-ready scorecard，因为仓库中的 `credit_visable.scoring.pdo_scorecard` 仍是 `placeholder`，当前也只做了 calibration diagnostics，没有拟合新的 calibration model。
- 报告怎么用：先看 `summary.json` 和 `score_transform_meta.json` 明确候选模型、分数参数和 placeholder 边界；再看 `risk_band_summary.csv`、`calibration_summary.csv`、`cutoff_sweep.csv`、`policy_scenarios.csv`；最后看 `phase6` 的 score / band / cutoff / migration 图，并把这些结果当成策略讨论输入，而不是直接当成生产配置。

## Reading Notes

- 当前 preprocessing pipeline 的 missing value 处理是基础插补，不是高级插补：数值变量使用 `median imputation`，类别变量使用 `most_frequent imputation`。
- 文档里需要区分两类变量：`EDA-derived analytical fields` 指 `Phase 1` 为解释和诊断构造的变量；`current modeling input fields` 指 `Phase 2-6` 当前真正进入主流程的 `application_train` 原始列及其编码结果。
- 文档里也需要区分三类结论：`descriptive finding` 用于描述数据现象；`model comparison finding` 用于比较数据口径或模型家族；`policy diagnostic finding` 用于讨论 score / cutoff / governance，不应直接宣称 production policy 已定稿。

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

项目默认从 `data/raw/` 读取 CSV。预期表名定义在 [`configs/base.yaml`](configs/base.yaml) 中，核心输入包括：

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

辅助文件如 `sample_submission.csv`、`HomeCredit_columns_description.csv` 可以放在 `data/raw/`，但不是默认 pipeline 的关键输入。

## Running in Colab

1. 先把当前本地修改推到 GitHub。
2. 在 Google Colab 中 clone 仓库。
3. 安装 package 依赖和可选 explainability / tree backend 依赖。
4. 如果数据放在 Google Drive，需要在 notebook 里手动设置 `RAW_DATA_DIR`。
5. 从 `notebooks/00_colab_setup.ipynb` 开始，按 `Phase 0 -> Phase 6` 顺序执行。

典型 Colab 初始化流程：

```python
!git clone https://github.com/GDSherlock/Make-Credit-Visable.git
%cd Make-Credit-Visable
!pip install -U pip
!pip install -r requirements.txt
!pip install -e .
!pip install xgboost shap
```

如果数据在 Google Drive，例如：

```python
from pathlib import Path
RAW_DATA_DIR = Path("/content/drive/MyDrive/home-credit/data/raw")
```

`RAW_DATA_DIR` 是运行时输入，不应把本地或 Drive 的会话路径提交回仓库。

## Artifact Map

当前 notebook workflow 会把标准产物写到这些目录：

- `data/processed/phase0_environment/`：环境、依赖和原始表可用性检查
- `data/processed/eda/`：主表 / 历史表 EDA summary、missingness、IV / WOE、interaction heatmap、分组切片
- `data/processed/preprocessing/`：双 feature-set preprocessing artifact、feature catalog 与 decision manifest
- `data/processed/modeling_baseline/`：logistic baseline comparison、data uplift、calibration / gain / lift diagnostics
- `data/processed/modeling_advanced/`：four-model comparison、model uplift、data uplift、best candidate summary、calibration / gain / lift diagnostics
- `data/processed/xai_fairness/`：candidate selection、explainability、interaction / PDP、grouped governance review、fairness metric summary
- `data/processed/scorecard_cutoff/`：score transform、risk band、calibration、cutoff、policy scenario、profit curve
- `reports/figures/`：各 phase 的导出图，文件前缀为 `phase1_` 到 `phase6_`

## Starter Modules

- `credit_visable.data`：raw table loading 与基础内存工具
- `credit_visable.features`：feature set 定义、preprocessing、IV / WOE placeholder
- `credit_visable.modeling`：baseline、advanced model、evaluation
- `credit_visable.explainability`：SHAP scaffold
- `credit_visable.governance`：grouped fairness summary scaffold
- `credit_visable.scoring`：scorecard / PDO placeholder
- `credit_visable.utils`：路径、绘图样式与通用工具

## Development Notes

- 尽量把路径逻辑收敛到 `credit_visable.utils.paths`。
- 尽量把可复用逻辑收敛到 `src/` 模块，而不是散落在 notebook 里。
- notebook 负责 phase-by-phase 的分析链路和报告导出，module 负责复用逻辑。
- 如果要扩展高级模型、explainability 或 fairness，请先确认当前 phase 产物和依赖已经完整。

## Status

当前仓库已经具备从 `Phase 0` 到 `Phase 6` 的 notebook-driven workflow，并且仓库中已经存在一套已落盘的 phase artifact，可直接作为方法和报告模板来阅读。

当前边界也需要明确：

- 多表客户级聚合还不是主流程
- preprocessing 仍以基础插补和 one-hot 为主
- `credit_visable.scoring.pdo_scorecard` 仍是 `placeholder`
- fairness 结果是 grouped governance diagnostic，不是完整 policy audit
