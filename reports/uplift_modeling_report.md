# Causal Uplift Modeling Report
- **Dataset:** `data\raw\credit_marketing.csv`
- **Treatment:** `marketing_offer`
- **Outcome:** `loan_application`
- **Customers scored:** 50,000
- **Output file:** `data\processed\customer_uplift_scores.csv`

## Individual Treatment Effect (ITE)

For each customer *i*, models estimate:

```
ITE(x_i) = E[Y | T=1, X=x_i] - E[Y | T=0, X=x_i]
```

where **Y** = loan application and **T** = marketing offer. Positive scores indicate
customers expected to apply **because of** the marketing intervention.

## Models Implemented (EconML)

| Model | EconML Class | Description |
| --- | --- | --- |
| S-Learner | `SLearner` | Single outcome model with treatment as feature |
| T-Learner | `TLearner` | Separate models for control and treated groups |
| X-Learner | `XLearner` | Imputed effects with propensity weighting |
| Causal Forest | `CausalForestDML` | DML + heterogeneous causal forest |

## ITE Distribution Summary

| Model | Mean ITE | Median ITE | Std | Min | Max | % Positive |
| --- | --- | --- | --- | --- | --- | --- |
| S-Learner | 0.0275 | 0.0252 | 0.0663 | -0.2320 | 0.2500 | 74.0% |
| T-Learner | 0.0289 | 0.0087 | 0.1397 | -0.5120 | 0.6215 | 54.8% |
| X-Learner | 0.0287 | 0.0018 | 0.1222 | -0.4316 | 0.4212 | 51.7% |
| Causal Forest | 0.0283 | 0.0002 | 0.1100 | -0.1666 | 0.2907 | 50.7% |

## Average Uplift Score by Segment

| Segment | Count | Avg Uplift | Median | Std |
| --- | --- | --- | --- | --- |
| lost_cause | 12,500 | 0.0073 | 0.0073 | 0.0170 |
| persuadable | 12,500 | 0.1825 | 0.1911 | 0.0539 |
| sleeping_dog | 12,500 | -0.0823 | -0.0780 | 0.0543 |
| sure_thing | 12,500 | 0.0060 | 0.0064 | 0.0311 |

## Output Columns

| Column | Description |
| --- | --- |
| `customer_id` | Customer identifier |
| `marketing_offer` | Observed treatment assignment |
| `loan_application` | Observed outcome |
| `uplift_s_learner` | ITE from S-Learner |
| `uplift_t_learner` | ITE from T-Learner |
| `uplift_x_learner` | ITE from X-Learner |
| `uplift_causal_forest` | ITE from Causal Forest |
| `uplift_score` | Ensemble average across all four models |

## Usage

Rank customers by `uplift_score` (or a model-specific column) to prioritize
marketing offers toward **persuadable** customers with high incremental conversion potential.
Exclude customers with negative predicted uplift (sleeping dogs).