# Data Quality Report — Credit Marketing Dataset

- **Dataset:** `data\raw\credit_marketing.csv`
- **Rows:** 50,000
- **Columns:** 17

## Executive Summary

**Overall status: PASS** — no critical issues detected.

## 1. Missing Values

| column | missing_count | missing_pct |
| --- | --- | --- |
| customer_id | 0 | 0.0000 |
| customer_segment | 0 | 0.0000 |
| age | 0 | 0.0000 |
| gender | 0 | 0.0000 |
| country | 0 | 0.0000 |
| income | 0 | 0.0000 |
| employment_years | 0 | 0.0000 |
| account_balance | 0 | 0.0000 |
| debt_ratio | 0 | 0.0000 |
| credit_score | 0 | 0.0000 |
| previous_loans | 0 | 0.0000 |
| payment_history | 0 | 0.0000 |
| credit_utilization | 0 | 0.0000 |
| transaction_frequency | 0 | 0.0000 |
| app_usage_frequency | 0 | 0.0000 |
| marketing_offer | 0 | 0.0000 |
| loan_application | 0 | 0.0000 |

## 2. Duplicate Records

- Fully duplicated rows: **0**
- Duplicate `customer_id` values: **0**

## 3. Data Types

| column | expected | actual | valid |
| --- | --- | --- | --- |
| customer_id | integer | int64 | True |
| customer_segment | string | object | True |
| age | numeric | float64 | True |
| gender | string | object | True |
| country | string | object | True |
| income | numeric | float64 | True |
| employment_years | numeric | float64 | True |
| account_balance | numeric | float64 | True |
| debt_ratio | numeric | float64 | True |
| credit_score | numeric | float64 | True |
| previous_loans | integer | int64 | True |
| payment_history | numeric | float64 | True |
| credit_utilization | numeric | float64 | True |
| transaction_frequency | numeric | float64 | True |
| app_usage_frequency | numeric | float64 | True |
| marketing_offer | binary | int64 | True |
| loan_application | binary | int64 | True |

## 4. Value Ranges

| column | min_bound | max_bound | actual_min | actual_max | below_min | above_max | violations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| age | 18.0000 | 100.0000 | 22.0000 | 65.0000 | 0 | 0 | 0 |
| income | 0.0000 | nan | 15000.0000 | 231517.7000 | 0 | 0 | 0 |
| employment_years | 0.0000 | nan | 0.0000 | 29.0000 | 0 | 0 | 0 |
| account_balance | 0.0000 | nan | 100.0000 | 138207.9400 | 0 | 0 | 0 |
| debt_ratio | 0.0000 | 1.0000 | 0.0230 | 0.8970 | 0 | 0 | 0 |
| credit_score | 300.0000 | 850.0000 | 341.0000 | 850.0000 | 0 | 0 | 0 |
| previous_loans | 0.0000 | nan | 0.0000 | 8.0000 | 0 | 0 | 0 |
| payment_history | 0.0000 | 1.0000 | 0.0040 | 1.0000 | 0 | 0 | 0 |
| credit_utilization | 0.0000 | 1.0000 | 0.0210 | 0.8980 | 0 | 0 | 0 |
| transaction_frequency | 0.0000 | nan | 0.0000 | 64.0000 | 0 | 0 | 0 |
| app_usage_frequency | 0.0000 | nan | 0.0000 | 47.0000 | 0 | 0 | 0 |

## 5. Outliers (IQR Method)

Values outside `[Q1 − 1.5×IQR, Q3 + 1.5×IQR]` are flagged as outliers.

| column | outlier_count | outlier_pct | lower_fence | upper_fence |
| --- | --- | --- | --- | --- |
| account_balance | 1577 | 3.1500 | -24962.5463 | 46052.1038 |
| app_usage_frequency | 1068 | 2.1400 | -12.5000 | 23.5000 |
| transaction_frequency | 636 | 1.2700 | -16.0000 | 32.0000 |
| income | 541 | 1.0800 | -33154.1925 | 129860.1475 |
| employment_years | 468 | 0.9400 | -5.0000 | 19.0000 |
| previous_loans | 216 | 0.4300 | -3.0000 | 5.0000 |
| age | 0 | 0.0000 | 10.0000 | 66.0000 |
| credit_score | 0 | 0.0000 | 314.5000 | 1070.5000 |
| debt_ratio | 0 | 0.0000 | -0.6095 | 1.4025 |
| credit_utilization | 0 | 0.0000 | -0.7090 | 1.5550 |
| payment_history | 0 | 0.0000 | -0.0825 | 1.4255 |

## 6. Treatment Balance

- Treated: **24,952** (49.90%)
- Control: **25,048** (50.10%)
- Binomial test vs 50% assignment: p = **0.6709**

### Treatment Rate by Segment

| customer_segment | n | treated | treatment_rate |
| --- | --- | --- | --- |
| lost_cause | 12500 | 6249 | 0.4999 |
| persuadable | 12500 | 6199 | 0.4959 |
| sleeping_dog | 12500 | 6220 | 0.4976 |
| sure_thing | 12500 | 6284 | 0.5027 |

## 7. Outcome Distribution

- Loan applications: **18,288** (36.58%)
- No application: **31,712** (63.42%)

### Response Rate by Treatment

| group | n | applications | response_rate |
| --- | --- | --- | --- |
| control | 25048 | 8770 | 0.3501 |
| treatment | 24952 | 9518 | 0.3815 |

### Response Rate by Segment

| customer_segment | n | applications | response_rate | control_rate | treatment_rate | uplift |
| --- | --- | --- | --- | --- | --- | --- |
| lost_cause | 12500 | 1614 | 0.1291 | 0.1326 | 0.1256 | -0.0070 |
| persuadable | 12500 | 5317 | 0.4254 | 0.3284 | 0.5240 | 0.1956 |
| sleeping_dog | 12500 | 4023 | 0.3218 | 0.3653 | 0.2780 | -0.0873 |
| sure_thing | 12500 | 7334 | 0.5867 | 0.5756 | 0.5977 | 0.0221 |
