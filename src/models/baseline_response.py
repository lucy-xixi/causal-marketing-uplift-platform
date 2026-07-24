"""Baseline models for P(loan_application) using customer attributes only."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.evaluation.metrics import ClassificationMetrics, evaluate_classifier
from src.features.preprocessing import (
    CUSTOMER_FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_preprocessor,
    prepare_model_data,
)


@dataclass(frozen=True)
class ModelResult:
    """Training and evaluation result for one baseline model."""

    name: str
    metrics: ClassificationMetrics
    train_size: int
    test_size: int
    positive_rate_train: float
    positive_rate_test: float


def _build_logistic_regression() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def _build_xgboost() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "model",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_and_evaluate_model(
    name: str,
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> tuple[ModelResult, np.ndarray]:
    """Fit a model and evaluate on the held-out test set."""
    pipeline.fit(x_train, y_train)
    y_prob = pipeline.predict_proba(x_test)[:, 1]
    metrics = evaluate_classifier(y_test.to_numpy(), y_prob, threshold=threshold)

    result = ModelResult(
        name=name,
        metrics=metrics,
        train_size=len(x_train),
        test_size=len(x_test),
        positive_rate_train=float(y_train.mean()),
        positive_rate_test=float(y_test.mean()),
    )
    return result, y_prob


def run_baseline_comparison(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    threshold: float = 0.5,
) -> tuple[list[ModelResult], pd.DataFrame, pd.Series, dict[str, np.ndarray]]:
    """Train baseline models and return comparison artifacts."""
    x, y = prepare_model_data(df)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    models = {
        "Logistic Regression": _build_logistic_regression(),
        "XGBoost": _build_xgboost(),
    }

    results: list[ModelResult] = []
    probabilities: dict[str, np.ndarray] = {}

    for name, pipeline in models.items():
        result, y_prob = train_and_evaluate_model(
            name=name,
            pipeline=pipeline,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            threshold=threshold,
        )
        results.append(result)
        probabilities[name] = y_prob

    return results, x_test, y_test, probabilities


def _format_pct(value: float) -> str:
    return f"{value:.2%}"


def _format_metric(value: float) -> str:
    return f"{value:.4f}"


def render_comparison_report(
    results: list[ModelResult],
    feature_columns: tuple[str, ...],
    dataset_path: Path,
    threshold: float,
    test_size: float,
    random_state: int,
) -> str:
    """Render a Markdown comparison report for baseline models."""
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    best_auc = max(results, key=lambda r: r.metrics.roc_auc)

    lines = [
        "# Baseline Response Prediction — Model Comparison Report",
        "",
        f"- **Generated:** {generated_at}",
        f"- **Dataset:** `{dataset_path}`",
        f"- **Target:** `{TARGET_COLUMN}` (probability of loan application)",
        f"- **Decision threshold:** {threshold:.2f}",
        f"- **Train/test split:** {1 - test_size:.0%} / {test_size:.0%} (stratified, seed={random_state})",
        "",
        "## Objective",
        "",
        "Predict **P(loan_application)** from **customer attributes only**. Treatment assignment",
        "(`marketing_offer`) and uplift segment labels are excluded — this baseline measures how",
        "well observable customer profiles alone explain conversion intent.",
        "",
        "## Features Used",
        "",
        "| Type | Columns |",
        "| --- | --- |",
        f"| Numeric ({len([c for c in feature_columns if c not in ('gender', 'country')])}) | "
        + ", ".join(c for c in feature_columns if c not in ("gender", "country"))
        + " |",
        "| Categorical (2) | gender, country |",
        "",
        "## Dataset Summary",
        "",
        f"- Training rows: **{results[0].train_size:,}** (positive rate: {_format_pct(results[0].positive_rate_train)})",
        f"- Test rows: **{results[0].test_size:,}** (positive rate: {_format_pct(results[0].positive_rate_test)})",
        "",
        "## Model Comparison",
        "",
        "| Model | ROC-AUC | Precision | Recall |",
        "| --- | --- | --- | --- |",
    ]

    for result in results:
        best_tag = " ← best AUC" if result.name == best_auc.name else ""
        lines.append(
            f"| {result.name}{best_tag} | "
            f"{_format_metric(result.metrics.roc_auc)} | "
            f"{_format_metric(result.metrics.precision)} | "
            f"{_format_metric(result.metrics.recall)} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- **Best ROC-AUC:** {best_auc.name} ({best_auc.metrics.roc_auc:.4f}) — strongest ranking performance.",
            "- **Precision** measures how many predicted applicants actually apply.",
            "- **Recall** measures how many actual applicants the model captures.",
            "- These models predict *who is likely to apply*, not the *incremental effect* of marketing.",
            "  Uplift modeling is required to estimate causal campaign impact.",
            "",
            "## Next Steps",
            "",
            "1. Tune decision thresholds for business-specific precision/recall trade-offs.",
            "2. Compare against propensity and uplift models that incorporate treatment.",
            "3. Evaluate calibration of predicted probabilities.",
        ]
    )

    return "\n".join(lines)


def save_roc_curve_plot(
    y_test: pd.Series,
    probabilities: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """Save an ROC curve comparison chart."""
    from sklearn.metrics import auc, roc_curve

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 5.5))

    for name, y_prob in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1)
    ax.set_title("ROC Curve — Baseline Response Models")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_baseline_pipeline(
    input_path: Path,
    report_path: Path,
    roc_plot_path: Path | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    threshold: float = 0.5,
) -> list[ModelResult]:
    """End-to-end baseline training, evaluation, and report generation."""
    df = pd.read_csv(input_path)
    results, _x_test, y_test, probabilities = run_baseline_comparison(
        df,
        test_size=test_size,
        random_state=random_state,
        threshold=threshold,
    )

    report = render_comparison_report(
        results=results,
        feature_columns=CUSTOMER_FEATURE_COLUMNS,
        dataset_path=input_path,
        threshold=threshold,
        test_size=test_size,
        random_state=random_state,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    if roc_plot_path is not None:
        save_roc_curve_plot(y_test, probabilities, roc_plot_path)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train baseline response prediction models and generate comparison report.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/credit_marketing.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/baseline_model_comparison.md"),
    )
    parser.add_argument(
        "--roc-plot",
        type=Path,
        default=Path("reports/baseline_model_roc.png"),
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    results = run_baseline_pipeline(
        input_path=args.input,
        report_path=args.output,
        roc_plot_path=args.roc_plot,
        test_size=args.test_size,
        random_state=args.random_state,
        threshold=args.threshold,
    )

    print(f"Report saved to {args.output.resolve()}")
    print(f"ROC plot saved to {args.roc_plot.resolve()}")
    print("\nModel comparison:")
    for result in results:
        m = result.metrics
        print(
            f"  {result.name:22s}  AUC={m.roc_auc:.4f}  "
            f"Precision={m.precision:.4f}  Recall={m.recall:.4f}"
        )


if __name__ == "__main__":
    main()
