# Project Overview Proposal (V3)

## 1. Introduction

This project focuses on the development of an application-stage credit risk scoring framework using the Home Credit Default Risk dataset. The central purpose of the project is to support more effective lending decisions by identifying applicants who are more likely to default and by translating model outputs into interpretable risk scores and rating bands that can be understood in a business setting. From a business perspective, the project aims to improve approval quality, reduce expected credit losses, strengthen portfolio segmentation, and support more consistent credit policy decisions. From a technical perspective, the project aims to build a customer-level analytical dataset from multiple related tables, engineer both traditional application features and application-time available historical behavioural and credit features, compare multiple predictive models, and evaluate model quality from the perspectives of discrimination, calibration, interpretability, fairness, and practical usability.

The project will adopt a staged modelling approach. It will begin with a baseline model built on traditional application-stage features, then extend the analysis by incorporating aggregated historical credit and behavioural features that would plausibly be available at the time of application. These behaviour-enhanced features are not intended to represent post-loan future information, but rather prior repayment, bureau, and credit usage histories summarised into customer-level indicators. The modelling workflow will compare interpretable scorecard-oriented approaches with stronger machine learning methods, and where appropriate, model outputs will be transformed into score bands using a PDO-based framework so that the final results can be presented in a form that is more aligned with credit risk practice.

---

## 2. Industry Overview

The project is situated in the consumer lending, retail banking, and fintech credit risk industry. Institutions operating in this space must continuously assess whether an applicant should be approved, rejected, priced differently, or referred for manual review. The quality of this assessment is fundamental because poor underwriting decisions can lead to higher default rates, declining portfolio quality, increased capital pressure, and weaker profitability, while overly conservative approval strategies may unnecessarily suppress business growth.

In practice, lenders traditionally rely on structured application data, bureau information, and historical repayment records to estimate default risk at the point of application. Over time, however, the industry has moved beyond purely static application variables and increasingly uses richer feature engineering, behavioural history summaries, machine learning models, and governance tools such as explainability and fairness diagnostics. This project reflects that broader industry direction. Although the dataset is primarily composed of traditional structured credit data, it allows the construction of behaviour-enhanced risk indicators from historical records, making it suitable for studying how more advanced feature engineering and model development can improve application-stage risk assessment.

---

## 3. Business Problem & Objectives

The core business problem is how to identify high-risk borrowers more accurately at the point of application so that lending decisions are commercially sound, operationally practical, and sufficiently transparent. This problem is critical because lenders must balance growth and risk at the same time. Approving too many risky borrowers can increase bad debt and impair portfolio performance, while rejecting too many potentially good borrowers can reduce revenue opportunities and weaken competitiveness.

Several business questions arise from this problem. First, the lender needs to know which applicants are most likely to default and whether they can be separated reliably from lower-risk borrowers before a loan is approved. Second, the lender needs to determine whether historical credit and behavioural information can improve risk prediction beyond what is available from static application variables alone. Third, the lender needs outputs that can be operationalised, such as risk scores, rating bands, and thresholds that could support approval, rejection, or manual review decisions. These questions are critical because they determine whether a model can move from an academic prediction exercise to a tool that supports real decision-making.

The business objectives of the project are therefore to improve applicant risk classification, reduce expected credit losses, support more disciplined loan approval decisions, and segment borrowers into categories that are meaningful for underwriting. The technical objectives are aligned to these goals. Technically, the project seeks to integrate multi-table credit data into a consistent customer-level modelling base, engineer risk-relevant features from application, bureau, and historical behaviour records, compare multiple modelling approaches, and evaluate their performance using predictive, interpretive, and governance-oriented metrics. This alignment is direct: better feature construction supports stronger risk discrimination, model comparison supports more informed model selection, score transformation supports business usability, and explainability and fairness analysis support governance and responsible deployment.

---

## 4. Project Design

The project will follow a staged road map designed to move from business understanding to model interpretation in a structured way. The first stage will focus on understanding the lending context, reviewing the dataset structure, identifying the analytical unit and target definition, and clarifying which data elements are appropriate for application-stage modelling. This is particularly important because the project aims to study A-score style modelling, which means that any historical behavioural enhancement should be framed as information that is available at the application decision point rather than post-loan future behaviour.

The second stage will involve exploratory data analysis. This will include examining the target distribution, identifying missing values and outliers, reviewing variable types, and understanding how the main application table relates to bureau, repayment, and balance-history tables. The goal of this stage is not only to understand the data statistically, but also to identify which variables and tables are likely to carry meaningful business signals.

The third stage will focus on data preparation and feature engineering. During this stage, one-to-many historical tables will be aggregated into customer-level summaries, categorical and numerical variables will be cleaned and transformed, and derived features such as ratios, trends, volatility indicators, repayment consistency measures, and utilisation metrics will be created. A distinction will be maintained between traditional application-stage features and behaviour-enhanced historical features so that model comparisons can be carried out in a structured way.

The fourth stage will involve model development. A baseline Logistic Regression model will be built first in order to provide an interpretable benchmark. This will then be compared with stronger machine learning approaches such as XGBoost or LightGBM, and an optional MLP may be explored as an experimental benchmark if the data preparation and sample size make that worthwhile. The fifth stage will focus on model evaluation and interpretation. Models will be compared using predictive metrics and business-oriented measures, and the contribution of key variables will be examined using explainability techniques such as SHAP. Calibration, fairness, and potential stability considerations will also be reviewed to determine whether any performance improvements are practically meaningful.

Finally, the last stage will translate technical outputs into business framing. Predicted probabilities may be converted into credit scores using a PDO-based transformation, and those scores may then be grouped into rating bands to support clearer business interpretation. This final stage is intended to connect the modelling exercise back to underwriting practice by demonstrating how a predictive model can be turned into a more usable decision-support framework.

---

## 5. Scope of Work

### Data Set

The project will use the Home Credit Default Risk dataset from Kaggle. The dataset is composed primarily of structured tabular credit data, including application information, external bureau records, historical loan applications, instalment payments, credit card balances, and POS or cash loan balances. In industry terms, this is largely traditional credit risk data, but its multi-table design also allows the creation of behaviour-enhanced features derived from prior credit usage and repayment histories.

The data preparation process is expected to include standard cleaning and transformation tasks such as handling missing values, reviewing outliers, encoding categorical variables, and aligning the analytical grain at the customer level. Because several tables are recorded at a transactional or monthly level, a major part of the work will involve aggregating one-to-many records into meaningful customer-level summaries. Additional transformations are likely to include creating ratio variables, trend and volatility indicators, repayment discipline measures, utilisation metrics, and selected proxy features that enrich the application-stage risk profile without relying on post-loan future information.

### Proposed Models, Techniques, and Evaluation Methodologies

The modelling approach will begin with Logistic Regression as the main interpretable baseline and then extend to more powerful machine learning models such as XGBoost or LightGBM. An MLP may be considered as an optional benchmark if a neural network comparison is useful for the project narrative. In addition to model fitting, the project may include scorecard-oriented techniques such as IV/WOE analysis where relevant, probability-to-score transformation using PDO, and explainability methods such as SHAP to understand the drivers of model predictions.

Evaluation will be carried out using both predictive and business-oriented methodologies. Predictive performance will be assessed using measures such as ROC-AUC, KS statistic, Precision, Recall, F1-score, and PR-AUC. Business relevance will be considered through score-band bad rates, lift or gain, and cutoff analysis to understand how model outputs might support lending decisions. Model quality will also be assessed through calibration analysis and fairness diagnostics, including metrics such as Disparate Impact and Equalized Odds where appropriate. If the data structure permits, some stability-oriented analysis may also be included to test how robust the model appears across different slices of the data.

---

## 6. Key Deliverables

The final outcomes of the project are expected to include a clearly articulated business and industry framing of the problem, a documented customer-level analytical dataset created from the multi-table source data, and a feature engineering framework that distinguishes between traditional application variables and behaviour-enhanced historical features. The project is also expected to deliver a baseline interpretable model, one or more stronger machine learning comparison models, and an evaluation report that summarises predictive performance, business usefulness, and model governance findings.

In addition, the project should produce explainability outputs, fairness diagnostics, and where feasible, a score development output that maps predicted default probabilities into interpretable risk scores and rating bands. The final deliverable should therefore not be limited to a single trained model, but should take the form of a complete project package containing the problem framing, technical workflow, modelling results, and practical conclusions that could support future extension into a more formal application scorecard development exercise.
