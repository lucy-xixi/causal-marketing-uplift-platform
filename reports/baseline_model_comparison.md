# Baseline Response Prediction — Model Comparison Report

- **Dataset:** `data\raw\credit_marketing.csv`
- **Target:** `loan_application` (probability of loan application)
- **Decision threshold:** 0.50
- **Train/test split:** 80% / 20% (stratified, seed=42)

## Objective

Predict **P(loan_application)** from **customer attributes only**. Treatment assignment
(`marketing_offer`) and uplift segment labels are excluded — this baseline measures how
well observable customer profiles alone explain conversion intent.

## Features Used

| Type | Columns |
| --- | --- |
| Numeric (11) | age, income, employment_years, account_balance, debt_ratio, credit_score, previous_loans, payment_history, credit_utilization, transaction_frequency, app_usage_frequency |
| Categorical (2) | gender, country |

## Dataset Summary

- Training rows: **40,000** (positive rate: 36.58%)
- Test rows: **10,000** (positive rate: 36.58%)

## Model Comparison

| Model | ROC-AUC | Precision | Recall |
| --- | --- | --- | --- |
| Logistic Regression | 0.7821 | 0.5615 | 0.7504 |
| XGBoost ← best AUC | 0.9080 | 0.7978 | 0.7387 |

## Interpretation

- **Best ROC-AUC:** XGBoost (0.9080) — strongest ranking performance.
- **Precision** measures how many predicted applicants actually apply.
- **Recall** measures how many actual applicants the model captures.
- These models predict *who is likely to apply*, not the *incremental effect* of marketing.
  Uplift modeling is required to estimate causal campaign impact.

## Next Steps

1. Tune decision thresholds for business-specific precision/recall trade-offs.
2. Compare against propensity and uplift models that incorporate treatment.
3. Evaluate calibration of predicted probabilities.