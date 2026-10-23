
# Causal Marketing Uplift Platform

An end-to-end causal uplift modeling prototype for a simulated personal-loan marketing scenario. The project compares response prediction with heterogeneous treatment-effect estimation to explore how marketing offers can be targeted toward customers who are more likely to respond *because of* the intervention.

> **Portfolio project:** All customer records and reported evaluation results are generated from synthetic data. The results demonstrate the modeling and evaluation workflow; they are not evidence of real-world marketing performance or financial returns.

## Business Problem

Traditional response models estimate which customers are likely to apply for a loan. Uplift modeling addresses a different question:

**Which customers are more likely to apply because they receive a marketing offer?**

This distinction matters because customers who are already highly likely to convert may not need an additional intervention. The goal is to estimate heterogeneous treatment effects and compare targeting strategies based on incremental response rather than response probability alone.

## Key Features

- **Synthetic data generation:** Creates a dataset of 50,000 simulated customers with four predefined response-effect segments: Persuadable, Sure Thing, Lost Cause, and Sleeping Dog.
- **Data quality assessment:** Checks missing values, duplicates, data types, value ranges, outliers, and treatment-group balance.
- **Exploratory analysis:** Provides a Jupyter Notebook for examining customer characteristics and treatment patterns.
- **Baseline response modeling:** Uses Logistic Regression and XGBoost to predict response.
- **Causal uplift modeling:** Uses EconML estimators, including S-Learner, T-Learner, X-Learner, and Causal Forest DML.
- **Targeting strategy evaluation:** Compares random targeting, response-model-based targeting, and uplift-based targeting using Qini and incremental-gain metrics.
- **Reporting:** Generates model comparison results, evaluation reports, and supporting figures.

## Modeling Approach

Let:

- \(T\) denote the marketing offer indicator (`marketing_offer`);
- \(Y\) denote the loan application outcome (`loan_application`);
- \(X\) denote customer characteristics.

The individual treatment effect is conceptually defined as:

\[
\tau(x) = E[Y \mid T=1, X=x] - E[Y \mid T=0, X=x]
\]

The project compares four uplift estimators:

| Estimator | Approach |
|---|---|
| S-Learner | Fits one outcome model with treatment as a feature |
| T-Learner | Fits separate outcome models for treatment and control groups |
| X-Learner | Estimates treatment effects using imputed effects and propensity scores |
| Causal Forest DML | Estimates heterogeneous effects using a causal forest and double machine learning |

The current `uplift_score` is an ensemble average of the estimated effects from the four models. These estimates are meaningful within the assumptions and synthetic data-generating process used by this demonstration.

## Evaluation Results

The current evaluation uses a synthetic test set of 10,000 customers.

| Targeting strategy | Qini coefficient | Incremental conversions in top 30% |
|---|---:|---:|
| Random targeting | 0.0 | 95.8 |
| XGBoost response model | 41.5 | 102.2 |
| EconML uplift ensemble | 131.9 | 481.8 |

In this simulation, the uplift strategy produces approximately 4.7 times the incremental conversions of response-model targeting within the top 30% of customers.

**Interpretation:** These figures illustrate how targeting based on estimated treatment effects can differ from targeting based on predicted response. They depend on the synthetic data and evaluation design and should not be interpreted as expected real-world lift, ROI, or financial performance.

## Technology Stack

- Python 3.11+
- pandas, NumPy, SciPy
- scikit-learn, XGBoost
- EconML
- Matplotlib, Seaborn
- Jupyter Notebook

## Repository Structure

```text
causal-marketing-uplift-platform/
├── configs/
│   └── default.yaml
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── 01_credit_marketing_eda.ipynb
├── reports/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   └── simulation/
├── requirements.txt
└── README.md
```

Key modules:

- `src/data/generate_credit_data.py`: Generates synthetic customer data.
- `src/data/assess_data_quality.py`: Produces data quality assessments.
- `src/features/preprocessing.py`: Handles feature selection and preprocessing.
- `src/models/baseline_response.py`: Trains and evaluates baseline response models.
- `src/models/uplift_models.py`: Fits uplift estimators and generates customer-level scores.
- `src/evaluation/uplift_evaluation.py`: Compares uplift curves, Qini metrics, and targeting strategies.
- `src/evaluation/metrics.py`: Provides classification and uplift evaluation metrics.

## Quick Start

### 1. Set up the environment

Clone the repository, then run the following commands from the project root:

```bash
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run the pipeline

Execute the steps in order:

```bash
# Generate 50,000 synthetic customer records
python src/data/generate_credit_data.py

# Assess data quality
python src/data/assess_data_quality.py

# Train and compare baseline response models
python -m src.models.baseline_response

# Fit uplift models and generate customer scores
python -m src.models.uplift_models

# Evaluate uplift and targeting strategies
python -m src.evaluation.uplift_evaluation
```

### 3. Explore the notebook

```bash
jupyter notebook notebooks/01_credit_marketing_eda.ipynb
```

## Generated Outputs

| Output | Description |
|---|---|
| `data/raw/credit_marketing.csv` | Synthetic customer dataset |
| `data/processed/customer_uplift_scores.csv` | Customer-level treatment-effect estimates |
| `reports/data_quality_report.md` | Data quality assessment |
| `reports/baseline_model_comparison.md` | Baseline model comparison |
| `reports/uplift_modeling_report.md` | Uplift modeling summary |
| `reports/uplift_evaluation_report.md` | Targeting strategy evaluation |
| `reports/uplift_evaluation_curves.png` | Uplift and Qini curves |
| `reports/uplift_incremental_gain.png` | Incremental-gain comparison |
| `reports/causal_uplift_project_report.md` | Detailed project report |

## Limitations

- The data is synthetic and does not represent real loan applicants or a real marketing campaign.
- The evaluation results depend on the assumptions embedded in the synthetic data generator.
- Estimated treatment effects are not directly transferable to a real population without appropriate data, study design, and validation.
- The prototype demonstrates an analytical workflow; it is not a deployed marketing decision system.

## License

See the repository's `LICENSE` file.
