# Project Overview Proposal

## 1. Introduction

### Project Summary
This project aims to develop an **application-stage credit risk scoring framework** using the **Home Credit Default Risk** dataset. The objective is to support more informed lending decisions by predicting the likelihood that a loan applicant will default and by transforming model outputs into interpretable risk scores and rating bands.

From a **business perspective**, the project seeks to help lenders:
- improve the quality of loan approval decisions,
- reduce credit losses from high-risk borrowers,
- improve portfolio risk segmentation,
- and support more consistent and explainable credit policies.

From a **technical perspective**, the project seeks to:
- construct a customer-level analytical dataset from multiple relational tables,
- engineer both traditional application features and behaviour-enhanced features,
- compare multiple machine learning and scorecard-oriented models,
- evaluate model performance, interpretability, fairness, and stability,
- and build a foundation for application scorecard development.

### Potential Approaches
Potential approaches for this project include:
- building a **baseline model** using traditional application features,
- enhancing the model with aggregated behavioural and historical credit features,
- comparing interpretable statistical models and more powerful machine learning models,
- and optionally converting predicted default probabilities into **credit scores** using a **PDO-based score transformation**.

---

## 2. Industry Overview

This project is situated in the **consumer lending / retail banking / fintech credit risk** industry.

In this industry, financial institutions must decide whether to:
- approve or reject a loan application,
- determine how risky an applicant is,
- assign internal risk grades,
- and potentially adjust pricing, credit limit, or manual review requirements.

Credit risk assessment is critical because poor underwriting decisions can lead to:
- higher default rates,
- increased capital pressure,
- weaker portfolio quality,
- and lower profitability.

Traditionally, lenders rely on:
- applicant demographic and financial information,
- bureau and repayment history,
- and scorecard-based decision systems.

However, modern credit risk modelling increasingly incorporates:
- behavioural risk indicators,
- richer feature engineering,
- machine learning methods,
- and model governance tools such as explainability and fairness checks.

This project reflects that industry trend by combining **traditional structured credit data** with **behaviour-enhanced engineered features** in an application scoring context.

---

## 3. Business Problem & Objectives

### 3.1 Business Problem
The core business problem is:

**How can a lender better identify high-risk applicants at the point of loan application, so that approval decisions are more accurate, explainable, and commercially sustainable?**

This problem matters because lenders must balance two competing priorities:
- approving enough customers to grow business,
- while avoiding excessive defaults and poor-quality lending.

### 3.2 Key Business Questions
Key business questions include:

1. **Which applicants are most likely to default?**  
   This is critical because the lender must identify risky borrowers before approving a loan.

2. **Can historical credit and behavioural information improve risk prediction beyond static application data?**  
   This is important because relying only on basic applicant information may understate risk.

3. **How can predicted risk be translated into operational lending decisions?**  
   Businesses need practical outputs such as risk scores, rating bands, and decision thresholds.

4. **How can the model remain interpretable, fair, and operationally credible?**  
   Credit decisions affect real people, so transparency and governance are important.

### 3.3 Business Objectives
The business objectives are to:
- improve applicant risk classification,
- support better loan approval decisions,
- reduce expected defaults and bad debt exposure,
- segment applicants into actionable risk bands,
- and provide interpretable model outputs for decision-making.

### 3.4 Technical Objectives
The technical objectives are to:
- integrate multi-table credit data into a customer-level modelling dataset,
- engineer meaningful risk features from application, bureau, and behavioural records,
- compare multiple predictive models,
- evaluate discrimination, calibration, stability, and fairness,
- and produce a scoring framework that can support scorecard-like deployment.

### 3.5 Alignment Between Business and Technical Objectives
The technical objectives directly support the business objectives:
- **Better feature engineering** supports more accurate risk prediction.
- **Model comparison** supports better model selection for business deployment.
- **Score transformation and risk banding** support practical lending use cases.
- **Explainability and fairness evaluation** support governance and responsible decision-making.

---

## 4. Project Design

### Proposed Road Map

#### Phase 1: Business Understanding and Data Familiarisation
- understand the lending use case,
- review the dataset structure and relationships,
- identify the target variable and analytical unit,
- determine which data is available at application stage.

#### Phase 2: Data Understanding and Exploratory Analysis
- inspect target distribution,
- assess missingness, outliers, and class imbalance,
- understand major feature groups,
- review relationships between application, bureau, and behavioural tables.

#### Phase 3: Data Preparation and Feature Engineering
- clean and standardise the data,
- aggregate historical tables to customer level,
- create ratio, trend, volatility, and behavioural features,
- separate traditional features from behaviour-enhanced features for model comparison.

#### Phase 4: Model Development
- build a baseline Logistic Regression model,
- develop stronger machine learning models such as XGBoost or LightGBM,
- optionally test MLP as an experimental benchmark.

#### Phase 5: Model Evaluation and Interpretation
- compare models using classification and business metrics,
- analyse feature importance and SHAP explanations,
- evaluate calibration and fairness metrics,
- assess whether behavioural enhancement adds value.

#### Phase 6: Score Development and Business Framing
- transform probabilities into risk scores,
- define rating bands,
- discuss potential approval / rejection / review thresholds,
- summarise findings in business and technical terms.

---

## 5. Scope of Work

### 5.1 Data Set

#### Source of Data
The project will use the **Home Credit Default Risk** dataset from Kaggle.

#### Type of Data
The dataset is primarily **structured tabular credit data**, including:
- application data,
- bureau / external credit history,
- historical applications,
- instalment payments,
- credit card balances,
- POS / cash loan balances.

This is mainly **traditional credit risk data**, but it also supports the creation of **behaviour-enhanced features** from historical and monthly performance records.

#### Potential Data Preparation and Transformation
Data preparation may include:
- handling missing values,
- treating outliers,
- encoding categorical variables,
- aggregating one-to-many historical tables to customer level,
- generating derived ratios and interaction terms,
- constructing behavioural metrics such as delinquency frequency, utilisation, repayment consistency, and trend-based indicators,
- splitting data for training, validation, and testing.

### 5.2 Proposed Models & Techniques

#### Proposed Models
- **Logistic Regression**  
  Interpretable baseline model.
- **XGBoost / LightGBM**  
  Strong predictive benchmark.
- **MLP (optional)**  
  Experimental deep learning benchmark for comparison.

#### Proposed Techniques
- feature aggregation across relational tables,
- ratio and proxy feature creation,
- optional IV / WOE analysis for scorecard-oriented modelling,
- SHAP for interpretability,
- fairness diagnostics,
- probability-to-score transformation using PDO.

### 5.3 Evaluation Methodologies

#### Predictive Performance
- ROC-AUC
- KS statistic
- Precision / Recall / F1
- PR-AUC

#### Business-Oriented Evaluation
- score band bad rates,
- lift / gain,
- cutoff analysis,
- risk segmentation performance.

#### Model Quality and Governance
- calibration analysis,
- SHAP explainability,
- fairness metrics such as:
  - Disparate Impact
  - Equalized Odds
- stability analysis (if time-based validation is feasible).

---

## 6. Key Deliverables

The final deliverables are expected to include:

1. **Project overview and business framing**  
   Problem statement, objectives, and industry context.

2. **Data understanding and preparation summary**  
   Description of data sources, structure, cleaning, and transformations.

3. **Customer-level modelling dataset**  
   Integrated analytical base table with engineered features.

4. **Feature engineering framework**  
   Traditional features, behaviour-enhanced features, and documented aggregation logic.

5. **Model development outputs**  
   Logistic Regression baseline, XGBoost / LightGBM comparison, and optional MLP benchmark.

6. **Model evaluation report**  
   Predictive metrics, business metrics, and model comparison results.

7. **Model governance outputs**  
   SHAP interpretation, fairness assessment, and calibration findings.

8. **Score development output**  
   Probability-to-score transformation and risk segmentation into rating bands.

9. **Final project report / presentation**  
   Business insights, technical approach, findings, limitations, and recommendations for future work.
