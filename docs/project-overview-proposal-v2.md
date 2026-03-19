# Project Overview Proposal v2

## 1. Introduction

This project aims to develop an application-stage credit risk scoring framework using the **Home Credit Default Risk** dataset. Its purpose is to support more effective lending decisions by estimating the probability that an applicant will default and by translating that probability into an interpretable score and risk band structure that can be used in practice. From a business standpoint, the project is intended to improve approval quality, reduce expected credit losses, strengthen portfolio segmentation, and support more consistent and explainable lending policies. From a technical standpoint, the project seeks to build a customer-level analytical base table from multiple related datasets, engineer both traditional application variables and history-based behavioural risk features, compare several modelling approaches, and evaluate performance, explainability, fairness, and calibration.

A key point in this project is that behavioural enhancement does **not** refer to using post-loan future behaviour from the current application. Instead, it refers to using **historical behavioural and credit information that would already be available at the application decision point**, such as prior repayment records, prior credit usage, bureau history, and previous application outcomes. These historical records can be aggregated into customer-level features and used to enrich the A-score model without violating the logic of application-stage scoring.

The project will therefore explore a progression from a traditional baseline model based on application features to an enhanced model that incorporates aggregated historical credit and behavioural features. In addition, it will examine whether the resulting outputs can be converted into an interpretable credit score using a PDO-based transformation and then grouped into practical rating bands for business use.

---

## 2. Industry Overview

This project is positioned within the **consumer lending, retail banking, and fintech credit risk** domain. In this industry, lenders must decide whether an applicant should be approved, how much risk that applicant represents, and how that risk should influence approval, pricing, limits, or referral for manual review. These decisions are central to profitability because poor underwriting can lead to elevated defaults, weaker portfolio quality, and greater capital and provisioning pressure, while overly conservative lending can reduce growth and market reach.

Traditional credit decisioning has historically relied on demographic information, employment and income characteristics, bureau records, and scorecard-based underwriting systems. More recently, the industry has moved toward richer feature engineering, the use of machine learning models alongside interpretable scorecards, and stronger model governance through explainability, fairness assessment, and calibration review. This project reflects that broader industry direction by combining traditional structured credit data with application-time available historical behavioural signals in order to improve risk estimation while preserving business interpretability.

---

## 3. Business Problem & Objectives

The central business problem is how a lender can more accurately identify high-risk applicants at the point of application so that loan approvals are commercially sound, operationally practical, and explainable to stakeholders. This is critical because lenders must continuously balance growth against risk: approving too many risky borrowers can increase defaults and credit losses, but rejecting too aggressively can suppress business expansion and reduce competitiveness.

Several business questions arise from this problem. First, the lender needs to know which applicants are most likely to default and whether that risk can be estimated reliably before approval. Second, the lender needs to determine whether historical credit and repayment behaviour can improve predictive power beyond static application information such as income, employment type, and declared financial profile. Third, the lender needs to understand how model outputs can be transformed into operationally useful tools such as credit scores, rating bands, and decision thresholds. Finally, the lender needs confidence that the resulting model is sufficiently interpretable, fair, and stable to support responsible decision-making.

The business objectives of the project are therefore to improve applicant risk classification, support better lending decisions, reduce expected bad debt exposure, and segment customers into actionable risk bands. The technical objectives are designed to align directly with these business needs. They include integrating multiple relational tables into a customer-level modelling dataset, engineering meaningful risk features from application, bureau, historical application, and repayment records, comparing interpretable and high-performance modelling approaches, and evaluating discrimination, calibration, fairness, and score usability. In this way, the technical work is not separate from the business objective but is the means by which better credit decisioning can be achieved.

---

## 4. Project Design

The project will follow a staged design that moves from business understanding into data familiarisation, feature engineering, modelling, and score interpretation. It will begin with a clear review of the lending use case, the structure of the dataset, the definition of the modelling unit, and the distinction between information available at application time and information that would only become visible after the loan decision. This distinction is especially important because the project aims to remain conceptually consistent with A-score development.

After that, the work will move into exploratory analysis. This stage will examine target distribution, missingness patterns, outliers, imbalance issues, and the structure of the major tables. It will also assess how the main application table connects with bureau history, prior applications, instalment behaviour, card balance records, and POS or cash loan records. From there, the focus will shift to data preparation and feature engineering. Historical tables will be aggregated to the customer level, and additional ratio, trend, volatility, and repayment-discipline variables will be created so that both traditional and behaviour-enhanced models can be tested.

Model development will then proceed in a comparative manner. A Logistic Regression model will serve as the interpretable baseline, while stronger predictive models such as XGBoost or LightGBM will be used as performance benchmarks. An MLP model may also be considered as an exploratory benchmark if time and model complexity allow. Once these models are trained, the next stage will evaluate them using predictive metrics, business-oriented segmentation logic, calibration analysis, and explainability tools such as SHAP. The final stage will examine the extent to which the predicted probabilities can be translated into a practical scoring scale, segmented into rating bands, and linked to possible approval, rejection, or manual review thresholds.

---

## 5. Scope of Work

The project will use the **Home Credit Default Risk** dataset from Kaggle. This dataset consists primarily of structured tabular credit risk data, including application information, external bureau records, historical applications, instalment payments, credit card balance records, and POS or cash loan balance records. In substantive terms, it represents traditional credit risk data rather than true external alternative data such as telecom, utility, geolocation, or device data. However, it does support the construction of behaviour-enhanced features by aggregating prior repayment and credit usage records that are assumed to be available at the point of application.

The data preparation process is expected to include handling missing values, treating outliers, encoding categorical variables, and converting one-to-many historical tables into customer-level aggregates. It will also include the creation of derived variables such as financial ratios, repayment discipline indicators, utilisation measures, volatility features, and other proxy variables that capture historical risk behaviour. These transformations will form the basis of the analytical dataset used for model development.

In terms of modelling, the project proposes to use Logistic Regression as the primary interpretable baseline, with XGBoost or LightGBM as stronger predictive alternatives. An optional MLP model may be explored as a comparison point, though it is not expected to be the primary business-facing model. Additional techniques may include IV or WOE analysis for scorecard-oriented modelling, SHAP for interpretability, fairness diagnostics, and a PDO-based probability-to-score conversion for score development.

Evaluation will be conducted using both predictive and business-relevant methodologies. Predictive discrimination may be assessed through ROC-AUC, KS, Precision, Recall, F1, and PR-AUC, while business usefulness may be examined through bad-rate separation across score bands, lift or gain analysis, and cutoff-based segmentation. Governance-oriented evaluation will consider calibration, fairness metrics such as Disparate Impact and Equalized Odds where appropriate, and stability analysis if the data structure permits time-based validation.

---

## 6. Key Deliverables

The final outcomes of the project are expected to include a clear statement of the business problem and project framing, a documented summary of the dataset and preparation process, and a customer-level modelling dataset containing engineered risk features. They will also include a documented feature engineering framework that distinguishes between traditional application variables and aggregated historical behavioural features, along with a comparative modelling output covering at least Logistic Regression and one or more machine learning benchmarks.

In addition, the project is expected to produce a model evaluation report that summarises predictive performance, business-facing segmentation quality, and comparative model findings. Model governance outputs such as SHAP-based interpretation, fairness diagnostics, and calibration review will also be included where feasible. Finally, the project aims to deliver a score development layer that converts predicted probabilities into interpretable risk scores and rating bands, supported by a final report or presentation that summarises the business rationale, technical methodology, results, limitations, and recommendations for future extension.
