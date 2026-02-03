"""Evaluation metrics for anomaly detection."""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class MetricsResult:
    """Container for evaluation metrics."""
    auroc: float
    auprc: float
    precision_at_k: Dict[int, float]
    recall_at_fpr: Dict[float, float]
    nan_rate: float


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k_values: list = None,
    fpr_values: list = None,
) -> MetricsResult:
    """Compute anomaly detection metrics.

    Args:
        y_true: Binary labels (1 = anomaly)
        y_score: Anomaly scores (higher = more anomalous)
        k_values: Values of k for precision@k
        fpr_values: FPR thresholds for recall@fpr

    Returns:
        MetricsResult with all computed metrics
    """
    if k_values is None:
        k_values = [50, 100, 200]
    if fpr_values is None:
        fpr_values = [0.01, 0.05, 0.10]

    # Check for NaN
    nan_mask = np.isnan(y_score)
    nan_rate = nan_mask.mean()

    if nan_rate == 1.0:
        return MetricsResult(
            auroc=float("nan"),
            auprc=float("nan"),
            precision_at_k={k: float("nan") for k in k_values},
            recall_at_fpr={fpr: float("nan") for fpr in fpr_values},
            nan_rate=nan_rate,
        )

    # Remove NaN for metric computation
    valid_mask = ~nan_mask
    y_true_valid = y_true[valid_mask]
    y_score_valid = y_score[valid_mask]

    # AUROC
    try:
        auroc = roc_auc_score(y_true_valid, y_score_valid)
    except ValueError:
        auroc = float("nan")

    # AUPRC
    try:
        auprc = average_precision_score(y_true_valid, y_score_valid)
    except ValueError:
        auprc = float("nan")

    # Precision@k
    precision_at_k = {}
    sorted_idx = np.argsort(y_score_valid)[::-1]
    for k in k_values:
        if k <= len(sorted_idx):
            top_k_labels = y_true_valid[sorted_idx[:k]]
            precision_at_k[k] = top_k_labels.mean()
        else:
            precision_at_k[k] = float("nan")

    # Recall@FPR
    recall_at_fpr = {}
    try:
        fpr, tpr, _ = roc_curve(y_true_valid, y_score_valid)
        for target_fpr in fpr_values:
            # Find threshold where FPR <= target_fpr
            idx = np.searchsorted(fpr, target_fpr)
            if idx < len(tpr):
                recall_at_fpr[target_fpr] = tpr[idx]
            else:
                recall_at_fpr[target_fpr] = tpr[-1]
    except ValueError:
        recall_at_fpr = {fpr: float("nan") for fpr in fpr_values}

    return MetricsResult(
        auroc=auroc,
        auprc=auprc,
        precision_at_k=precision_at_k,
        recall_at_fpr=recall_at_fpr,
        nan_rate=nan_rate,
    )


def evaluate_model(
    model,
    x_test: np.ndarray,
    c_test: np.ndarray,
    y_test: np.ndarray,
    model_type: str = "cwae",
) -> MetricsResult:
    """Evaluate a model on test data.

    Args:
        model: Trained model with score() method
        x_test: Test features
        c_test: Test context
        y_test: Test labels
        model_type: "cwae", "cvae", "if", or "dif"

    Returns:
        MetricsResult
    """
    import torch

    if model_type in ["cwae", "cvae"]:
        x_tensor = torch.tensor(x_test, dtype=torch.float32)
        c_tensor = torch.tensor(c_test, dtype=torch.float32)
        scores = model.score(x_tensor, c_tensor)
    else:
        scores = model.score(x_test, c_test)

    return compute_metrics(y_test, scores)


def print_metrics(results: MetricsResult, name: str = "Model"):
    """Pretty print metrics."""
    print(f"\n{name} Results:")
    print(f"  AUROC: {results.auroc:.4f}")
    print(f"  AUPRC: {results.auprc:.4f}")
    print(f"  NaN Rate: {results.nan_rate:.2%}")
    print("  Precision@k:")
    for k, v in results.precision_at_k.items():
        print(f"    P@{k}: {v:.4f}")
    print("  Recall@FPR:")
    for fpr, v in results.recall_at_fpr.items():
        print(f"    R@{fpr:.0%}: {v:.4f}")
