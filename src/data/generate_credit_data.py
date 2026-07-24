"""Synthetic credit marketing dataset generator with heterogeneous treatment effects."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEGMENTS = ("persuadable", "sure_thing", "lost_cause", "sleeping_dog")

COUNTRIES = ("US", "UK", "DE", "FR", "ES", "IT", "NL", "CA")
GENDERS = ("Male", "Female", "Other")

# Segment-specific causal parameters: log-odds intercept and treatment lift.
SEGMENT_CAUSAL_PARAMS: dict[str, dict[str, float]] = {
    "persuadable": {"intercept": -1.4, "treatment_effect": 1.6},
    "sure_thing": {"intercept": 0.6, "treatment_effect": 0.12},
    "lost_cause": {"intercept": -3.6, "treatment_effect": 0.03},
    "sleeping_dog": {"intercept": -1.1, "treatment_effect": -0.85},
}

FEATURE_COEFFICIENTS = {
    "age": 0.015,
    "income_log": 0.35,
    "employment_years": 0.04,
    "account_balance_log": 0.12,
    "debt_ratio": -1.8,
    "credit_score_scaled": 1.2,
    "previous_loans": 0.18,
    "payment_history": 0.9,
    "credit_utilization": -0.7,
    "transaction_frequency_log": 0.25,
    "app_usage_frequency_log": 0.3,
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _clip_round(arr: np.ndarray, low: float, high: float, decimals: int = 0) -> np.ndarray:
    return np.round(np.clip(arr, low, high), decimals)


def _assign_segments(n: int, rng: np.random.Generator) -> np.ndarray:
    """Assign each customer to one of four uplift segments (balanced)."""
    segment_idx = np.arange(n) % len(SEGMENTS)
    rng.shuffle(segment_idx)
    return np.array(SEGMENTS)[segment_idx]


def _generate_demographics(n: int, rng: np.random.Generator) -> pd.DataFrame:
    age = _clip_round(rng.normal(38, 11, n), 22, 65)
    gender = rng.choice(GENDERS, size=n, p=[0.48, 0.48, 0.04])
    country = rng.choice(COUNTRIES, size=n, p=[0.35, 0.15, 0.12, 0.10, 0.08, 0.08, 0.06, 0.06])
    return pd.DataFrame({"age": age, "gender": gender, "country": country})


def _generate_financial(
    n: int,
    segments: np.ndarray,
    age: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    income = np.empty(n)
    employment_years = np.empty(n)
    account_balance = np.empty(n)
    debt_ratio = np.empty(n)

    for segment in SEGMENTS:
        mask = segments == segment
        count = mask.sum()
        if count == 0:
            continue

        if segment == "persuadable":
            income[mask] = rng.lognormal(mean=10.6, sigma=0.35, size=count)
            employment_years[mask] = _clip_round(rng.normal(6, 3, count), 1, 25)
            account_balance[mask] = rng.lognormal(mean=8.2, sigma=0.6, size=count)
            debt_ratio[mask] = rng.beta(5, 4, size=count) * 0.65 + 0.15
        elif segment == "sure_thing":
            income[mask] = rng.lognormal(mean=11.2, sigma=0.3, size=count)
            employment_years[mask] = _clip_round(rng.normal(10, 4, count), 2, 30)
            account_balance[mask] = rng.lognormal(mean=9.5, sigma=0.5, size=count)
            debt_ratio[mask] = rng.beta(3, 6, size=count) * 0.45 + 0.05
        elif segment == "lost_cause":
            income[mask] = rng.lognormal(mean=9.8, sigma=0.45, size=count)
            employment_years[mask] = _clip_round(rng.normal(3, 2.5, count), 0, 15)
            account_balance[mask] = rng.lognormal(mean=6.5, sigma=0.7, size=count)
            debt_ratio[mask] = rng.beta(7, 2.5, size=count) * 0.55 + 0.35
        else:  # sleeping_dog
            income[mask] = rng.lognormal(mean=11.0, sigma=0.35, size=count)
            employment_years[mask] = _clip_round(rng.normal(12, 5, count), 3, 35)
            account_balance[mask] = rng.lognormal(mean=10.2, sigma=0.45, size=count)
            debt_ratio[mask] = rng.beta(2.5, 7, size=count) * 0.35 + 0.02

    employment_years = np.minimum(employment_years, age - 18)

    return pd.DataFrame(
        {
            "income": _clip_round(income, 15_000, 250_000, decimals=2),
            "employment_years": employment_years,
            "account_balance": _clip_round(account_balance, 100, 500_000, decimals=2),
            "debt_ratio": _clip_round(debt_ratio, 0.0, 0.95, decimals=3),
        }
    )


def _generate_credit(
    n: int,
    segments: np.ndarray,
    income: np.ndarray,
    debt_ratio: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    credit_score = np.empty(n)
    previous_loans = np.empty(n, dtype=int)
    payment_history = np.empty(n)
    credit_utilization = np.empty(n)

    for segment in SEGMENTS:
        mask = segments == segment
        count = mask.sum()
        if count == 0:
            continue

        if segment == "persuadable":
            credit_score[mask] = rng.normal(680, 35, count)
            previous_loans[mask] = rng.poisson(1.2, count)
            payment_history[mask] = rng.beta(6, 2.5, size=count)
            credit_utilization[mask] = rng.beta(5, 4, size=count) * 0.75 + 0.15
        elif segment == "sure_thing":
            credit_score[mask] = rng.normal(780, 25, count)
            previous_loans[mask] = rng.poisson(2.0, count)
            payment_history[mask] = rng.beta(8, 1.5, size=count)
            credit_utilization[mask] = rng.beta(3, 6, size=count) * 0.55 + 0.05
        elif segment == "lost_cause":
            credit_score[mask] = rng.normal(540, 40, count)
            previous_loans[mask] = rng.poisson(0.8, count)
            payment_history[mask] = rng.beta(2.5, 6, size=count)
            credit_utilization[mask] = rng.beta(7, 2.5, size=count) * 0.45 + 0.45
        else:  # sleeping_dog
            credit_score[mask] = rng.normal(760, 30, count)
            previous_loans[mask] = rng.poisson(1.0, count)
            payment_history[mask] = rng.beta(7, 2, size=count)
            credit_utilization[mask] = rng.beta(2, 8, size=count) * 0.35 + 0.02

    credit_score = _clip_round(credit_score, 300, 850)
    previous_loans = np.clip(previous_loans, 0, 8)
    payment_history = _clip_round(payment_history, 0.0, 1.0, decimals=3)
    credit_utilization = _clip_round(credit_utilization, 0.0, 0.99, decimals=3)

    # Mild global correlation: higher income supports better credit profiles.
    credit_score += (income - income.mean()) / income.std() * 8
    credit_score -= (debt_ratio - debt_ratio.mean()) / debt_ratio.std() * 12
    credit_score = _clip_round(credit_score, 300, 850)

    return pd.DataFrame(
        {
            "credit_score": credit_score,
            "previous_loans": previous_loans,
            "payment_history": payment_history,
            "credit_utilization": credit_utilization,
        }
    )


def _generate_behavior(
    n: int,
    segments: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    transaction_frequency = np.empty(n)
    app_usage_frequency = np.empty(n)

    for segment in SEGMENTS:
        mask = segments == segment
        count = mask.sum()
        if count == 0:
            continue

        if segment == "persuadable":
            transaction_frequency[mask] = rng.gamma(4.0, 3.0, count)
            app_usage_frequency[mask] = rng.gamma(3.5, 2.5, count)
        elif segment == "sure_thing":
            transaction_frequency[mask] = rng.gamma(5.0, 3.5, count)
            app_usage_frequency[mask] = rng.gamma(4.5, 3.0, count)
        elif segment == "lost_cause":
            transaction_frequency[mask] = rng.gamma(2.0, 2.0, count)
            app_usage_frequency[mask] = rng.gamma(1.5, 1.5, count)
        else:  # sleeping_dog
            transaction_frequency[mask] = rng.gamma(1.2, 1.5, count)
            app_usage_frequency[mask] = rng.gamma(0.8, 1.2, count)

    return pd.DataFrame(
        {
            "transaction_frequency": _clip_round(transaction_frequency, 0, 120),
            "app_usage_frequency": _clip_round(app_usage_frequency, 0, 90),
        }
    )


def _standardize(arr: np.ndarray) -> np.ndarray:
    return (arr - arr.mean()) / arr.std()


def _generate_outcome(
    df: pd.DataFrame,
    segments: np.ndarray,
    treatment: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate loan_application via segment-specific causal logit models."""
    log_odds = np.zeros(len(df))

    for segment in SEGMENTS:
        mask = segments == segment
        params = SEGMENT_CAUSAL_PARAMS[segment]
        segment_log_odds = params["intercept"]

        segment_log_odds += FEATURE_COEFFICIENTS["age"] * _standardize(df.loc[mask, "age"].to_numpy())
        segment_log_odds += FEATURE_COEFFICIENTS["income_log"] * _standardize(
            np.log(df.loc[mask, "income"].to_numpy())
        )
        segment_log_odds += FEATURE_COEFFICIENTS["employment_years"] * _standardize(
            df.loc[mask, "employment_years"].to_numpy()
        )
        segment_log_odds += FEATURE_COEFFICIENTS["account_balance_log"] * _standardize(
            np.log1p(df.loc[mask, "account_balance"].to_numpy())
        )
        segment_log_odds += FEATURE_COEFFICIENTS["debt_ratio"] * _standardize(
            df.loc[mask, "debt_ratio"].to_numpy()
        )
        segment_log_odds += FEATURE_COEFFICIENTS["credit_score_scaled"] * _standardize(
            df.loc[mask, "credit_score"].to_numpy()
        )
        segment_log_odds += FEATURE_COEFFICIENTS["previous_loans"] * _standardize(
            df.loc[mask, "previous_loans"].to_numpy()
        )
        segment_log_odds += FEATURE_COEFFICIENTS["payment_history"] * _standardize(
            df.loc[mask, "payment_history"].to_numpy()
        )
        segment_log_odds += FEATURE_COEFFICIENTS["credit_utilization"] * _standardize(
            df.loc[mask, "credit_utilization"].to_numpy()
        )
        segment_log_odds += FEATURE_COEFFICIENTS["transaction_frequency_log"] * _standardize(
            np.log1p(df.loc[mask, "transaction_frequency"].to_numpy())
        )
        segment_log_odds += FEATURE_COEFFICIENTS["app_usage_frequency_log"] * _standardize(
            np.log1p(df.loc[mask, "app_usage_frequency"].to_numpy())
        )

        segment_log_odds += params["treatment_effect"] * treatment[mask]
        log_odds[mask] = segment_log_odds

    prob = _sigmoid(log_odds)
    return rng.binomial(1, prob).astype(int)


def generate_credit_marketing_data(
    n_customers: int = 50_000,
    treatment_rate: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic credit marketing dataset with heterogeneous treatment effects."""
    rng = np.random.default_rng(seed)

    customer_id = np.arange(1, n_customers + 1)
    segments = _assign_segments(n_customers, rng)

    demographics = _generate_demographics(n_customers, rng)
    financial = _generate_financial(n_customers, segments, demographics["age"].to_numpy(), rng)
    credit = _generate_credit(
        n_customers,
        segments,
        financial["income"].to_numpy(),
        financial["debt_ratio"].to_numpy(),
        rng,
    )
    behavior = _generate_behavior(n_customers, segments, rng)

    marketing_offer = rng.binomial(1, treatment_rate, n_customers).astype(int)

    features = pd.concat([demographics, financial, credit, behavior], axis=1)
    loan_application = _generate_outcome(features, segments, marketing_offer, rng)

    return pd.DataFrame(
        {
            "customer_id": customer_id,
            "customer_segment": segments,
            **features.to_dict(orient="list"),
            "marketing_offer": marketing_offer,
            "loan_application": loan_application,
        }
    )


def print_dataset_summary(df: pd.DataFrame) -> None:
    """Print treatment rate, response rate, and segment distribution."""
    treatment_rate = df["marketing_offer"].mean()
    response_rate = df["loan_application"].mean()

    print("\n=== Credit Marketing Dataset Summary ===")
    print(f"Customers:        {len(df):,}")
    print(f"Treatment rate:   {treatment_rate:.2%}")
    print(f"Response rate:    {response_rate:.2%}")

    print("\nSegment distribution:")
    segment_counts = df["customer_segment"].value_counts().sort_index()
    for segment, count in segment_counts.items():
        print(f"  {segment:14s} {count:6,}  ({count / len(df):.1%})")

    print("\nResponse rate by segment and treatment:")
    pivot = df.groupby(["customer_segment", "marketing_offer"])["loan_application"].mean().unstack()
    pivot.columns = ["control", "treatment"]
    pivot["uplift"] = pivot["treatment"] - pivot["control"]
    print(pivot.round(3).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic credit marketing data.")
    parser.add_argument("--n-customers", type=int, default=50_000)
    parser.add_argument("--treatment-rate", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/credit_marketing.csv"),
    )
    args = parser.parse_args()

    df = generate_credit_marketing_data(
        n_customers=args.n_customers,
        treatment_rate=args.treatment_rate,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved dataset to {args.output.resolve()}")
    print_dataset_summary(df)


if __name__ == "__main__":
    main()
