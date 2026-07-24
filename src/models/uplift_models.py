"""Causal uplift modeling with EconML meta-learners and causal forest."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from econml.metalearners import SLearner, TLearner, XLearner
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.features.preprocessing import (
    TARGET_COLUMN,
    TREATMENT_COLUMN,
    build_preprocessor,
    prepare_uplift_data,
)


class UpliftEstimator(Protocol):
    """Protocol for EconML-style uplift estimators."""

    def fit(self, y: np.ndarray, t: np.ndarray, *, X: np.ndarray) -> object: ...

    def effect(self, X: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class UpliftModelSpec:
    """Specification for an uplift learner."""

    name: str
    column_name: str
    builder: Callable[[], UpliftEstimator]


def _build_outcome_model() -> GradientBoostingRegressor:
    """Base regression model for outcome surfaces E[Y | X, T]."""
    return GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )


def build_s_learner() -> SLearner:
    """Single-model learner: E[Y | X, T] with T as input feature."""
    return SLearner(overall_model=_build_outcome_model())


def build_t_learner() -> TLearner:
    """Two-model learner: separate E[Y | X, T=0] and E[Y | X, T=1]."""
    return TLearner(models=_build_outcome_model())


def build_x_learner() -> XLearner:
    """X-Learner with propensity-weighted imputation of treatment effects."""
    return XLearner(
        models=_build_outcome_model(),
        cate_models=_build_outcome_model(),
        propensity_model=LogisticRegression(max_iter=1000, random_state=42),
    )


def build_causal_forest() -> CausalForestDML:
    """Double ML causal forest for heterogeneous treatment effects."""
    return CausalForestDML(
        model_y=GradientBoostingClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
        ),
        model_t=LogisticRegression(max_iter=1000, random_state=42),
        discrete_treatment=True,
        discrete_outcome=True,
        n_estimators=500,
        min_samples_leaf=50,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    )


UPLIFT_MODELS: tuple[UpliftModelSpec, ...] = (
    UpliftModelSpec("S-Learner", "uplift_s_learner", build_s_learner),
    UpliftModelSpec("T-Learner", "uplift_t_learner", build_t_learner),
    UpliftModelSpec("X-Learner", "uplift_x_learner", build_x_learner),
    UpliftModelSpec("Causal Forest", "uplift_causal_forest", build_causal_forest),
)


def estimate_ite(
    estimator: UpliftEstimator,
    y: np.ndarray,
    t: np.ndarray,
    x_train: np.ndarray,
    x_score: np.ndarray,
) -> np.ndarray:
    """
    Fit an uplift model and estimate individual treatment effects.

    ITE(x) = E[Y | T=1, X=x] - E[Y | T=0, X=x]
    """
    estimator.fit(y, t, X=x_train)
    ite = estimator.effect(x_score)
    return np.asarray(ite).reshape(-1)


def fit_all_uplift_models(
    x: pd.DataFrame,
    y: np.ndarray,
    t: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Train all uplift models on a hold-out split and score ITE for every customer.

    Returns
    -------
    scores_df : pd.DataFrame
        Customer-level uplift scores for each model.
    test_predictions : dict[str, np.ndarray]
        ITE predictions on the test split (for diagnostics).
    test_idx : np.ndarray
        Positional indices of test rows in the original dataframe.
    x_matrix : np.ndarray
        Preprocessed feature matrix for all customers.
    """
    indices = np.arange(len(x))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=t,
    )

    preprocessor = build_preprocessor()
    x_matrix = preprocessor.fit_transform(x)
    x_train = x_matrix[train_idx]
    y_train = y[train_idx]
    t_train = t[train_idx]

    score_columns: dict[str, np.ndarray] = {}
    test_predictions: dict[str, np.ndarray] = {}

    for spec in UPLIFT_MODELS:
        estimator = spec.builder()
        ite_all = estimate_ite(estimator, y_train, t_train, x_train, x_matrix)
        ite_test = estimator.effect(x_matrix[test_idx])

        score_columns[spec.column_name] = ite_all
        test_predictions[spec.name] = np.asarray(ite_test).reshape(-1)

    scores_df = pd.DataFrame(score_columns)
    return scores_df, test_predictions, test_idx, x_matrix


def build_customer_uplift_output(
    df: pd.DataFrame,
    uplift_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Combine identifiers, observed data, and model ITE scores."""
    output = pd.DataFrame(
        {
            "customer_id": df["customer_id"].values,
            TREATMENT_COLUMN: df[TREATMENT_COLUMN].values,
            TARGET_COLUMN: df[TARGET_COLUMN].values,
        }
    )

    if "customer_segment" in df.columns:
        output["customer_segment"] = df["customer_segment"].values

    for column in uplift_scores.columns:
        output[column] = uplift_scores[column].values

    # Primary score: average across learners for targeting convenience.
    uplift_cols = [spec.column_name for spec in UPLIFT_MODELS]
    output["uplift_score"] = output[uplift_cols].mean(axis=1)
    return output


def summarize_uplift_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Summary statistics for each model's ITE distribution."""
    rows = []
    for spec in UPLIFT_MODELS:
        values = scores[spec.column_name]
        rows.append(
            {
                "model": spec.name,
                "mean_ite": values.mean(),
                "median_ite": values.median(),
                "std_ite": values.std(),
                "min_ite": values.min(),
                "max_ite": values.max(),
                "pct_positive": (values > 0).mean(),
            }
        )
    summary = pd.DataFrame(rows)
    return summary.round(4)


def summarize_uplift_by_segment(
    output_df: pd.DataFrame,
    score_column: str = "uplift_score",
) -> pd.DataFrame | None:
    """Average predicted uplift by ground-truth segment (when available)."""
    if "customer_segment" not in output_df.columns:
        return None

    grouped = (
        output_df.groupby("customer_segment")[score_column]
        .agg(["count", "mean", "median", "std"])
        .rename(columns={"mean": "avg_uplift", "median": "median_uplift"})
        .round(4)
    )
    return grouped


def render_uplift_report(
    output_df: pd.DataFrame,
    summary: pd.DataFrame,
    dataset_path: Path,
    output_path: Path,
    scores_path: Path,
) -> str:
    """Render Markdown report for uplift modeling run."""
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    segment_summary = summarize_uplift_by_segment(output_df)

    lines = [
        "# Causal Uplift Modeling Report",
        "",
        f"- **Generated:** {generated_at}",
        f"- **Dataset:** `{dataset_path}`",
        f"- **Treatment:** `{TREATMENT_COLUMN}`",
        f"- **Outcome:** `{TARGET_COLUMN}`",
        f"- **Customers scored:** {len(output_df):,}",
        f"- **Output file:** `{scores_path}`",
        "",
        "## Individual Treatment Effect (ITE)",
        "",
        "For each customer *i*, models estimate:",
        "",
        "```",
        "ITE(x_i) = E[Y | T=1, X=x_i] - E[Y | T=0, X=x_i]",
        "```",
        "",
        "where **Y** = loan application and **T** = marketing offer. Positive scores indicate",
        "customers expected to apply **because of** the marketing intervention.",
        "",
        "## Models Implemented (EconML)",
        "",
        "| Model | EconML Class | Description |",
        "| --- | --- | --- |",
        "| S-Learner | `SLearner` | Single outcome model with treatment as feature |",
        "| T-Learner | `TLearner` | Separate models for control and treated groups |",
        "| X-Learner | `XLearner` | Imputed effects with propensity weighting |",
        "| Causal Forest | `CausalForestDML` | DML + heterogeneous causal forest |",
        "",
        "## ITE Distribution Summary",
        "",
        "| Model | Mean ITE | Median ITE | Std | Min | Max | % Positive |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    for _, row in summary.iterrows():
        lines.append(
            f"| {row['model']} | {row['mean_ite']:.4f} | {row['median_ite']:.4f} | "
            f"{row['std_ite']:.4f} | {row['min_ite']:.4f} | {row['max_ite']:.4f} | "
            f"{row['pct_positive']:.1%} |"
        )

    if segment_summary is not None:
        lines.extend(["", "## Average Uplift Score by Segment", ""])
        lines.append("| Segment | Count | Avg Uplift | Median | Std |")
        lines.append("| --- | --- | --- | --- | --- |")
        for segment, row in segment_summary.iterrows():
            lines.append(
                f"| {segment} | {int(row['count']):,} | {row['avg_uplift']:.4f} | "
                f"{row['median_uplift']:.4f} | {row['std']:.4f} |"
            )

    lines.extend(
        [
            "",
            "## Output Columns",
            "",
            "| Column | Description |",
            "| --- | --- |",
            "| `customer_id` | Customer identifier |",
            f"| `{TREATMENT_COLUMN}` | Observed treatment assignment |",
            f"| `{TARGET_COLUMN}` | Observed outcome |",
            "| `uplift_s_learner` | ITE from S-Learner |",
            "| `uplift_t_learner` | ITE from T-Learner |",
            "| `uplift_x_learner` | ITE from X-Learner |",
            "| `uplift_causal_forest` | ITE from Causal Forest |",
            "| `uplift_score` | Ensemble average across all four models |",
            "",
            "## Usage",
            "",
            "Rank customers by `uplift_score` (or a model-specific column) to prioritize",
            "marketing offers toward **persuadable** customers with high incremental conversion potential.",
            "Exclude customers with negative predicted uplift (sleeping dogs).",
        ]
    )

    markdown = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    return markdown


def run_uplift_pipeline(
    input_path: Path,
    scores_path: Path,
    report_path: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    """End-to-end uplift modeling: train, score ITE, save outputs."""
    df = pd.read_csv(input_path)
    x, t, y = prepare_uplift_data(df)

    uplift_scores, _test_preds, _test_idx, _x_matrix = fit_all_uplift_models(
        x,
        y,
        t,
        test_size=test_size,
        random_state=random_state,
    )

    output_df = build_customer_uplift_output(df, uplift_scores)
    summary = summarize_uplift_scores(uplift_scores)

    scores_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(scores_path, index=False)

    render_uplift_report(
        output_df=output_df,
        summary=summary,
        dataset_path=input_path,
        output_path=report_path,
        scores_path=scores_path,
    )

    return output_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EconML uplift models and score customer ITE.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/credit_marketing.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/customer_uplift_scores.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/uplift_modeling_report.md"),
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    output_df = run_uplift_pipeline(
        input_path=args.input,
        scores_path=args.output,
        report_path=args.report,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print(f"Customer uplift scores saved to {args.output.resolve()}")
    print(f"Report saved to {args.report.resolve()}")
    print(f"\nScored {len(output_df):,} customers")
    print("\nUplift score preview:")
    preview_cols = ["customer_id", "uplift_score", "uplift_s_learner", "uplift_t_learner",
                    "uplift_x_learner", "uplift_causal_forest"]
    print(output_df[preview_cols].head(10).to_string(index=False))
    print("\nAverage ITE by model:")
    for spec in UPLIFT_MODELS:
        mean_ite = output_df[spec.column_name].mean()
        print(f"  {spec.name:16s}  mean ITE = {mean_ite:+.4f}")


if __name__ == "__main__":
    main()
