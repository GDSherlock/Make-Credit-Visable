# Project References

This document lists potential references relevant to the Home Credit application-stage credit risk scoring project. The references are grouped by purpose so they can support the proposal, methodology, modelling choices, interpretability, fairness discussion, and broader business framing.

## 1. Dataset and Project Context

1. **Kaggle. (2018). Home Credit Default Risk.**  
   Available at: <https://www.kaggle.com/c/home-credit-default-risk>  
   This is the primary source of the dataset used in the project. It provides the business motivation, competition objective, task framing, and data tables used for modelling.

2. **Kaggle. Home Credit Default Risk – Data Description.**  
   Available at: <https://www.kaggle.com/c/home-credit-default-risk/data>  
   This is useful for understanding the structure of the application, bureau, repayment, and balance-related tables.

## 2. Credit Scoring and Scorecard Foundations

3. **Siddiqi, N. (2006). Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring. Hoboken, NJ: Wiley.**  
   This is one of the most practical references for scorecard development, covering scorecard logic, variable treatment, model validation, implementation, and monitoring.

4. **Thomas, L. C., Edelman, D. B., & Crook, J. N. (2002). Credit Scoring and Its Applications. Philadelphia, PA: SIAM.**  
   A core reference on credit scoring methodology and its applications in consumer lending and risk decision-making.

5. **Anderson, R. (2007). The Credit Scoring Toolkit: Theory and Practice for Retail Credit Risk Management and Decision Automation. Oxford: Oxford University Press.**  
   A useful practical reference on retail credit scoring, score development, and decision automation.

6. **Thomas, L. C. (2009). Consumer Credit Models: Pricing, Profit, and Portfolios. Oxford: Oxford University Press.**  
   Helpful for linking credit scoring methodology to portfolio management and business decision-making.

## 3. Machine Learning and Benchmarking in Credit Risk

7. **Lessmann, S., Baesens, B., Seow, H.-V., & Thomas, L. C. (2015). Benchmarking state-of-the-art classification algorithms for credit scoring: An update of research. European Journal of Operational Research, 247(1), 124–136.**  
   Available via abstract/indexing pages such as: <https://ideas.repec.org/a/eee/ejores/v247y2015i1p124-136.html>  
   This is one of the most widely cited benchmark papers comparing modern classification algorithms for credit scoring.

8. **Baesens, B., Van Gestel, T., Stepanova, M., Van den Poel, D., & Vanthienen, J. (2003). Neural network survival analysis for personal loan data. Journal of the Operational Research Society, 56(9), 1089–1098.**  
   A classic example of more advanced modelling approaches being applied in consumer credit settings.

9. **Bussmann, N., Giudici, P., Marinelli, D., & Papenbrock, J. (2021). Explainable machine learning in credit risk management. Computational Economics, 57, 203–216.**  
   Useful for connecting machine learning model use with explainability requirements in credit risk applications.

## 4. Logistic Regression, WOE, IV, and Score Transformation

10. **Siddiqi, N. (2006). Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring.**  
    Particularly relevant for WOE, IV, scorecard construction, and score-to-odds / PDO transformation.

11. **Anderson, R. (2007). The Credit Scoring Toolkit.**  
    Useful for score interpretation, score scaling, and deployment-oriented scorecard logic.

## 5. Explainability and Model Interpretation

12. **Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. Advances in Neural Information Processing Systems, 30.**  
    Available at: <https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html>  
    This is the standard SHAP reference and is highly relevant for explaining complex credit risk models.

13. **Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). “Why Should I Trust You?”: Explaining the Predictions of Any Classifier. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.**  
    Available at: <https://dl.acm.org/doi/10.1145/2939672.2939778>  
    The standard reference for LIME, useful if local explanation is discussed in the project.

14. **Molnar, C. (2022). Interpretable Machine Learning (2nd ed.).**  
    Available at: <https://christophm.github.io/interpretable-ml-book/>  
    A practical open reference for explainability concepts including SHAP, feature importance, and local explanations.

## 6. Fairness and Responsible Credit Risk Modelling

15. **Hardt, M., Price, E., & Srebro, N. (2016). Equality of Opportunity in Supervised Learning. Advances in Neural Information Processing Systems, 29.**  
    Available at: <https://proceedings.neurips.cc/paper/2016/hash/9d2682367c3935defcb1f9e247a97c0d-Abstract.html>  
    This is the canonical reference behind Equalized Odds and Equality of Opportunity.

16. **Barocas, S., Hardt, M., & Narayanan, A. (2023). Fairness and Machine Learning: Limitations and Opportunities.**  
    Available at: <https://fairmlbook.org/>  
    A widely used open reference for fairness concepts, trade-offs, and practical interpretation.

17. **Feldman, M., Friedler, S. A., Moeller, J., Scheidegger, C., & Venkatasubramanian, S. (2015). Certifying and Removing Disparate Impact. Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.**  
    Available at: <https://dl.acm.org/doi/10.1145/2783258.2783311>  
    Useful for understanding Disparate Impact in a machine learning context.

## 7. Alternative Data and Behavioural Features in Credit Risk

18. **Jagtiani, J., & Lemieux, C. (2019). The roles of alternative data and machine learning in fintech lending: evidence from the LendingClub consumer platform. Financial Management, 48(4), 1009–1029.**  
    Available via publisher/index pages such as SSRN summaries and journal pages.  
    This is useful for motivating the role of non-traditional and alternative-like variables in lending decisions.

19. **Berg, T., Burg, V., Gombović, A., & Puri, M. (2020). On the rise of fintechs: Credit scoring using digital footprints. Review of Financial Studies, 33(7), 2845–2897.**  
    This is a strong reference for the broader alternative-data discussion, even if the current project only uses proxy alternative features rather than true digital-footprint data.

20. **Hurley, M., & Adebayo, J. (2017). Credit scoring in the era of big data. Yale Journal of Law and Technology, 18, 148–216.**  
    A useful conceptual reference for the intersection of credit scoring, large-scale data, and fairness concerns.

## 8. How These References Support the Project

These references collectively support several parts of the project. The Kaggle references justify the dataset and application-stage default prediction problem. Siddiqi, Thomas, and Anderson support the scorecard and credit scoring foundations. Lessmann and related machine learning references support the comparison of Logistic Regression, XGBoost or LightGBM, and optionally MLP. Lundberg & Lee and Ribeiro et al. support the explainability components of the project, while Hardt et al., Barocas et al., and Feldman et al. support fairness analysis. Finally, Jagtiani & Lemieux and Berg et al. are useful for framing the discussion of non-traditional or proxy alternative features, even though the present project does not directly use external telecom or utility data.
