"""Data quality assessment for credit marketing datasets."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

EXPECTED_SEGMENTS = frozenset({"persuadable", "sure_thing", "lost_cause", "sleeping_dog"})
EXPECTED_GENDERS = frozenset({"Male", "Female", "Other"})
EXPECTED_COUNTRIES = frozenset({"US", "UK", "DE", "FR", "ES", "IT", "NL", "CA"})
EXPECTED_BINARY = frozenset({0, 1})

EXPECTED_DTYPES: dict[str, str] = {
    "customer_id": "integer",
    "customer_segment": "string",
    "age": "numeric",
    "gender": "string",
    "country": "string",
    "income": "numeric",
    "employment_years": "numeric",
    "account_balance": "numeric",
    "debt_ratio": "numeric",
    "credit_score": "numeric",
    "previous_loans": "integer",
    "payment_history": "numeric",
    "credit_utilization": "numeric",
    "transaction_frequency": "numeric",
    "app_usage_frequency": "numeric",
    "marketing_offer": "binary",
    "loan_application": "binary",
}

VALUE_RANGES: dict[str, dict[str, float | int | None]] = {
    "age": {"min": 18, "max": 100},
    "income": {"min": 0, "max": None},
    "employment_years": {"min": 0, "max": None},
    "account_balance": {"min": 0, "max": None},
    "debt_ratio": {"min": 0.0, "max": 1.0},
    "credit_score": {"min": 300, "max": 850},
    "previous_loans": {"min": 0, "max": None},
    "payment_history": {"min": 0.0, "max": 1.0},
    "credit_utilization": {"min": 0.0, "max": 1.0},
    "transaction_frequency": {"min": 0, "max": None},
    "app_usage_frequency": {"min": 0, "max": None},
}

NUMERIC_OUTLIER_COLUMNS = [
    "age",
    "income",
    "employment_years",
    "account_balance",
    "debt_ratio",
    "credit_score",
    "previous_loans",
    "payment_history",
    "credit_utilization",
    "transaction_frequency",
    "app_usage_frequency",
]


@dataclass
class QualityReport:
    """Container for data quality assessment results."""

    dataset_path: Path
    row_count: int
    column_count: int
    missing_values: pd.DataFrame = field(default_factory=pd.DataFrame)
    duplicate_rows: int = 0
    duplicate_customer_ids: int = 0
    dtype_checks: pd.DataFrame = field(default_factory=pd.DataFrame)
    range_violations: pd.DataFrame = field(default_factory=pd.DataFrame)
    invalid_categories: dict[str, list[str]] = field(default_factory=dict)
    outliers: pd.DataFrame = field(default_factory=pd.DataFrame)
    treatment_summary: pd.Series = field(default_factory=pd.Series)
    treatment_by_segment: pd.DataFrame = field(default_factory=pd.DataFrame)
    treatment_balance_pvalue: float = float("nan")
    outcome_summary: pd.Series = field(default_factory=pd.Series)
    outcome_by_treatment: pd.DataFrame = field(default_factory=pd.DataFrame)
    outcome_by_segment: pd.DataFrame = field(default_factory=pd.DataFrame)
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _is_integer_series(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        return False
    return np.allclose(numeric, np.floor(numeric))


def _is_binary_series(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        return False
    return set(numeric.unique()).issubset(EXPECTED_BINARY)


def _classify_dtype(series: pd.Series, expected: str) -> tuple[str, bool]:
    if expected == "integer":
        ok = pd.api.types.is_integer_dtype(series) or _is_integer_series(series)
        actual = str(series.dtype)
    elif expected == "numeric":
        ok = pd.api.types.is_numeric_dtype(series)
        actual = str(series.dtype)
    elif expected == "binary":
        ok = _is_binary_series(series)
        actual = str(series.dtype)
    else:
        ok = pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)
        actual = str(series.dtype)
    return actual, ok


def _detect_outliers_iqr(series: pd.Series) -> pd.Series:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (series < lower) | (series > upper)


def assess_data_quality(
    df: pd.DataFrame,
    dataset_path: Path,
    expected_treatment_rate: float = 0.5,
) -> QualityReport:
    """Run all data quality checks and return a structured report."""
    report = QualityReport(
        dataset_path=dataset_path,
        row_count=len(df),
        column_count=len(df.columns),
    )

    required_columns = list(EXPECTED_DTYPES.keys())
    missing_columns = [col for col in required_columns if col not in df.columns]
    extra_columns = [col for col in df.columns if col not in required_columns]

    if missing_columns:
        report.issues.append(f"Missing required columns: {', '.join(missing_columns)}")
    if extra_columns:
        report.warnings.append(f"Unexpected extra columns: {', '.join(extra_columns)}")

    # Missing values
    missing_count = df.isna().sum()
    missing_pct = (missing_count / len(df) * 100).round(4)
    report.missing_values = pd.DataFrame(
        {"missing_count": missing_count, "missing_pct": missing_pct}
    ).sort_values("missing_count", ascending=False)

    if missing_count.sum() > 0:
        cols_with_missing = missing_count[missing_count > 0].index.tolist()
        report.issues.append(f"Missing values detected in: {', '.join(cols_with_missing)}")

    # Duplicates
    report.duplicate_rows = int(df.duplicated().sum())
    if "customer_id" in df.columns:
        report.duplicate_customer_ids = int(df["customer_id"].duplicated().sum())
        if report.duplicate_customer_ids > 0:
            report.issues.append(
                f"Duplicate customer_id values: {report.duplicate_customer_ids:,}"
            )
    if report.duplicate_rows > 0:
        report.issues.append(f"Fully duplicated rows: {report.duplicate_rows:,}")

    # Data types
    dtype_rows = []
    for column, expected in EXPECTED_DTYPES.items():
        if column not in df.columns:
            continue
        actual, ok = _classify_dtype(df[column], expected)
        dtype_rows.append(
            {"column": column, "expected": expected, "actual": actual, "valid": ok}
        )
        if not ok:
            report.issues.append(
                f"Invalid data type for '{column}': expected {expected}, got {actual}"
            )
    report.dtype_checks = pd.DataFrame(dtype_rows)

    # Value ranges
    range_rows = []
    for column, bounds in VALUE_RANGES.items():
        if column not in df.columns:
            continue
        series = pd.to_numeric(df[column], errors="coerce")
        below = int((series < bounds["min"]).sum()) if bounds["min"] is not None else 0
        above = int((series > bounds["max"]).sum()) if bounds["max"] is not None else 0
        violations = below + above
        range_rows.append(
            {
                "column": column,
                "min_bound": bounds["min"],
                "max_bound": bounds["max"],
                "actual_min": series.min(),
                "actual_max": series.max(),
                "below_min": below,
                "above_max": above,
                "violations": violations,
            }
        )
        if violations > 0:
            report.issues.append(
                f"Range violations in '{column}': {violations:,} values outside bounds"
            )
    report.range_violations = pd.DataFrame(range_rows)

    # Categorical validity
    category_checks = {
        "customer_segment": EXPECTED_SEGMENTS,
        "gender": EXPECTED_GENDERS,
        "country": EXPECTED_COUNTRIES,
        "marketing_offer": EXPECTED_BINARY,
        "loan_application": EXPECTED_BINARY,
    }
    for column, allowed in category_checks.items():
        if column not in df.columns:
            continue
        invalid = sorted(set(df[column].dropna().unique()) - allowed)
        if invalid:
            report.invalid_categories[column] = [str(v) for v in invalid]
            report.issues.append(f"Invalid values in '{column}': {invalid}")

    # Outliers (IQR)
    outlier_rows = []
    for column in NUMERIC_OUTLIER_COLUMNS:
        if column not in df.columns:
            continue
        series = pd.to_numeric(df[column], errors="coerce")
        mask = _detect_outliers_iqr(series)
        count = int(mask.sum())
        outlier_rows.append(
            {
                "column": column,
                "outlier_count": count,
                "outlier_pct": round(count / len(df) * 100, 2),
                "lower_fence": series.quantile(0.25) - 1.5 * (series.quantile(0.75) - series.quantile(0.25)),
                "upper_fence": series.quantile(0.75) + 1.5 * (series.quantile(0.75) - series.quantile(0.25)),
            }
        )
    report.outliers = pd.DataFrame(outlier_rows).sort_values("outlier_count", ascending=False)

    high_outlier_cols = report.outliers[report.outliers["outlier_pct"] > 5.0]["column"].tolist()
    if high_outlier_cols:
        report.warnings.append(
            f"Columns with >5% IQR outliers: {', '.join(high_outlier_cols)}"
        )

    # Treatment balance
    if "marketing_offer" in df.columns:
        treated = df["marketing_offer"].sum()
        treatment_rate = treated / len(df)
        report.treatment_summary = pd.Series(
            {
                "treated_count": int(treated),
                "control_count": int(len(df) - treated),
                "treatment_rate": round(treatment_rate, 4),
            }
        )

        n_treated = int(treated)
        binom_result = stats.binomtest(n_treated, len(df), expected_treatment_rate)
        report.treatment_balance_pvalue = float(binom_result.pvalue)

        if "customer_segment" in df.columns:
            segment_counts = df.groupby("customer_segment")["marketing_offer"].agg(["count", "sum"])
            segment_counts["treatment_rate"] = segment_counts["sum"] / segment_counts["count"]
            report.treatment_by_segment = segment_counts.rename(
                columns={"count": "n", "sum": "treated"}
            )

        if report.treatment_balance_pvalue < 0.01:
            report.warnings.append(
                f"Treatment rate ({treatment_rate:.2%}) differs from expected "
                f"{expected_treatment_rate:.0%} (binom p={report.treatment_balance_pvalue:.4f})"
            )

    # Outcome distribution
    if "loan_application" in df.columns:
        responders = df["loan_application"].sum()
        response_rate = responders / len(df)
        report.outcome_summary = pd.Series(
            {
                "positive_count": int(responders),
                "negative_count": int(len(df) - responders),
                "response_rate": round(response_rate, 4),
            }
        )

        if "marketing_offer" in df.columns:
            outcome_treatment = df.groupby("marketing_offer")["loan_application"].agg(["count", "sum", "mean"])
            outcome_treatment.index = outcome_treatment.index.map({0: "control", 1: "treatment"})
            outcome_treatment.columns = ["n", "applications", "response_rate"]
            report.outcome_by_treatment = outcome_treatment

        if "customer_segment" in df.columns:
            outcome_segment = df.groupby("customer_segment")["loan_application"].agg(["count", "sum", "mean"])
            outcome_segment.columns = ["n", "applications", "response_rate"]
            if "marketing_offer" in df.columns:
                uplift = df.groupby(["customer_segment", "marketing_offer"])["loan_application"].mean().unstack()
                uplift.columns = ["control_rate", "treatment_rate"]
                uplift["uplift"] = uplift["treatment_rate"] - uplift["control_rate"]
                outcome_segment = outcome_segment.join(uplift[["control_rate", "treatment_rate", "uplift"]])
            report.outcome_by_segment = outcome_segment

    return report


def _format_table(df: pd.DataFrame, float_cols: int = 4) -> str:
    formatted = df.copy()
    for col in formatted.select_dtypes(include="number").columns:
        formatted[col] = formatted[col].apply(
            lambda x: f"{x:.{float_cols}f}" if pd.notna(x) and isinstance(x, float) else x
        )
    headers = "| " + " | ".join(formatted.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(formatted.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in formatted.to_numpy()]
    return "\n".join([headers, separator, *rows])


def _status_line(report: QualityReport) -> str:
    if report.issues:
        return f"**Overall status: FAIL** — {len(report.issues)} issue(s) detected."
    if report.warnings:
        return f"**Overall status: PASS WITH WARNINGS** — {len(report.warnings)} warning(s)."
    return "**Overall status: PASS** — no critical issues detected."


def render_markdown_report(report: QualityReport) -> str:
    """Render the quality report as Markdown."""
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Data Quality Report — Credit Marketing Dataset",
        "",
        f"- **Generated:** {generated_at}",
        f"- **Dataset:** `{report.dataset_path}`",
        f"- **Rows:** {report.row_count:,}",
        f"- **Columns:** {report.column_count}",
        "",
        "## Executive Summary",
        "",
        _status_line(report),
        "",
    ]

    if report.issues:
        lines.extend(["### Issues", ""])
        lines.extend(f"- {issue}" for issue in report.issues)
        lines.append("")

    if report.warnings:
        lines.extend(["### Warnings", ""])
        lines.extend(f"- {warning}" for warning in report.warnings)
        lines.append("")

    lines.extend(
        [
            "## 1. Missing Values",
            "",
            _format_table(report.missing_values.reset_index().rename(columns={"index": "column"})),
            "",
            "## 2. Duplicate Records",
            "",
            f"- Fully duplicated rows: **{report.duplicate_rows:,}**",
            f"- Duplicate `customer_id` values: **{report.duplicate_customer_ids:,}**",
            "",
            "## 3. Data Types",
            "",
            _format_table(report.dtype_checks),
            "",
            "## 4. Value Ranges",
            "",
            _format_table(report.range_violations),
            "",
            "## 5. Outliers (IQR Method)",
            "",
            "Values outside `[Q1 − 1.5×IQR, Q3 + 1.5×IQR]` are flagged as outliers.",
            "",
            _format_table(report.outliers),
            "",
            "## 6. Treatment Balance",
            "",
        ]
    )

    if not report.treatment_summary.empty:
        ts = report.treatment_summary
        lines.extend(
            [
                f"- Treated: **{int(ts['treated_count']):,}** ({ts['treatment_rate']:.2%})",
                f"- Control: **{int(ts['control_count']):,}** ({1 - ts['treatment_rate']:.2%})",
                f"- Binomial test vs 50% assignment: p = **{report.treatment_balance_pvalue:.4f}**",
                "",
            ]
        )
        if not report.treatment_by_segment.empty:
            lines.extend(
                [
                    "### Treatment Rate by Segment",
                    "",
                    _format_table(report.treatment_by_segment.reset_index()),
                    "",
                ]
            )

    lines.extend(["## 7. Outcome Distribution", ""])

    if not report.outcome_summary.empty:
        os_ = report.outcome_summary
        lines.extend(
            [
                f"- Loan applications: **{int(os_['positive_count']):,}** ({os_['response_rate']:.2%})",
                f"- No application: **{int(os_['negative_count']):,}** ({1 - os_['response_rate']:.2%})",
                "",
            ]
        )
        if not report.outcome_by_treatment.empty:
            lines.extend(
                [
                    "### Response Rate by Treatment",
                    "",
                    _format_table(report.outcome_by_treatment.reset_index().rename(columns={"marketing_offer": "group"})),
                    "",
                ]
            )
        if not report.outcome_by_segment.empty:
            lines.extend(
                [
                    "### Response Rate by Segment",
                    "",
                    _format_table(report.outcome_by_segment.reset_index()),
                    "",
                ]
            )

    return "\n".join(lines)


def run_quality_assessment(
    input_path: Path,
    output_path: Path,
    expected_treatment_rate: float = 0.5,
) -> QualityReport:
    """Load data, assess quality, and write Markdown report."""
    df = pd.read_csv(input_path)
    report = assess_data_quality(df, input_path, expected_treatment_rate)
    markdown = render_markdown_report(report)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Assess credit marketing data quality.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/credit_marketing.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/data_quality_report.md"),
    )
    parser.add_argument("--expected-treatment-rate", type=float, default=0.5)
    args = parser.parse_args()

    report = run_quality_assessment(args.input, args.output, args.expected_treatment_rate)
    print(f"Quality report saved to {args.output.resolve()}")
    print(_status_line(report))


if __name__ == "__main__":
    main()
