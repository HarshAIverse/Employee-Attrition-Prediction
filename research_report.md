# HR Analytics Research Report: Predictive Attrition Modeling

## Executive Summary
Palo Alto Networks has experienced unexpectedly high rates of resignation, particularly within certain technical and support departments. This report summarizes the findings of the Machine Learning modeling phase and provides insights derived from historical datasets. By switching from reactive retention to proactive scoring, HR can actively mitigate turnover.

## Problem Context
The current HR framework is predominantly descriptive. Without a predictive layer, managers lose transparency into the subtle combination of risk vectors. Overwork (OverTime), poor job satisfaction, and lacking managerial relations often compound silently.

## Exploratory Data Analysis & Strategic Discoveries
Through the initial modeling phases and Exploratory Data Analysis (EDA) on the HR dataset, several key patterns emerged:
1. **The Overtime Risk Multiplier**: Employees actively logging overtime are 3.5x more likely to leave the firm compared to their standard-hour counterparts.
2. **Promotion Delays**: Our algorithm heavily weights the `Promotion Delay Flag` (defined as 3+ years since last promotion). Employees in technical fields who stagnate in title have a significantly higher flight risk.
3. **Manager Relationship Score**: When cross-referencing Years with Current Manager and Relationship Satisfaction, low stability indices accurately forecast a trailing attrition event within 6 months.

## Methodology & Machine Learning Architecture
To counteract data imbalance (approx. 84% retained to 16% resigned), the Synthethic Minority Over-sampling Technique (SMOTE) was applied. This allowed the models to learn resignation profiles comprehensively rather than defaulting to "Retained" guesses.
We benchmarked:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost

**Conclusion**: XGBoost and Random Forest architectures proved most robust, yielding the highest F1 Scores without sacrificing Recall. A false positive (flagging a safe employee as an attrition risk) is a far better business outcome than a false negative (failing to flag someone who resigns).

## ROI and Recommendations
Implementing the Attrition Risk Engine Dashboard allows HR Business Partners to:
1. Identify high-risk individuals before a resignation letter is drafted.
2. Conduct "What-If" scenarios targeting specifically at-risk employees (e.g., modeling if a 15% salary hike would move an employee from "High Risk" to "Low Risk").
3. Uncover institutional toxicities in specific departments.

## Next Steps
Rollout access to the interactive Streamlit dashboard to all HR Executives. Following a 3-month observation phase, we will recalibrate the predictive baseline.
