"""Classification metrics for model evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class ClassificationMetrics:
    """Binary classification metrics at a fixed decision threshold."""

    roc_auc: float
    precision: float
    recall: float
    threshold: float = 0.5

    def as_dict(self) -> dict[str, float]:
        return {
            "roc_auc": self.roc_auc,
            "precision": self.precision,
            "recall": self.recall,
            "threshold": self.threshold,
        }


def evaluate_classifier(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> ClassificationMetrics:
    """Compute ROC-AUC, precision, and recall for predicted probabilities."""
    y_pred = (y_prob >= threshold).astype(int)
    return ClassificationMetrics(
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        threshold=threshold,
    )


def evaluate_average_precision(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Optional auxiliary ranking metric."""
    return float(average_precision_score(y_true, y_prob))
