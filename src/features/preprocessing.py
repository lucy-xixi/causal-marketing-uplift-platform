"""Feature preprocessing for response prediction models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "loan_application"
TREATMENT_COLUMN = "marketing_offer"
ID_COLUMNS = ("customer_id",)
EXCLUDED_COLUMNS = (
    "customer_segment",
    TREATMENT_COLUMN,
    TARGET_COLUMN,
)

NUMERIC_FEATURES = (
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
)

CATEGORICAL_FEATURES = (
    "gender",
    "country",
)

CUSTOMER_FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for customer-attribute feature matrices."""

    numeric_features: tuple[str, ...] = NUMERIC_FEATURES
    categorical_features: tuple[str, ...] = CATEGORICAL_FEATURES
    target_column: str = TARGET_COLUMN


def select_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return customer-attribute columns used for response prediction."""
    return df[list(CUSTOMER_FEATURE_COLUMNS)].copy()


def build_preprocessor(config: FeatureConfig | None = None) -> ColumnTransformer:
    """Build a sklearn preprocessor for numeric and categorical attributes."""
    config = config or FeatureConfig()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(config.numeric_features)),
            ("categorical", categorical_pipeline, list(config.categorical_features)),
        ]
    )


def prepare_model_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix X and binary target y from a raw dataframe."""
    missing = set(CUSTOMER_FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required feature columns: {sorted(missing)}")

    x = select_customer_features(df)
    y = df[TARGET_COLUMN].astype(int)
    return x, y


def prepare_uplift_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Extract X, treatment T, and outcome Y for causal uplift modeling."""
    missing = set(CUSTOMER_FEATURE_COLUMNS + (TREATMENT_COLUMN, TARGET_COLUMN)) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    x = select_customer_features(df)
    y = df[TARGET_COLUMN].astype(int).to_numpy()
    t = df[TREATMENT_COLUMN].astype(int).to_numpy()
    return x, t, y
