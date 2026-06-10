"""Advanced classification metric helpers.

Purpose:
    Compute the project-level metrics used in model comparison tables.

Inputs:
    Ground-truth labels, predicted labels, and optional predicted probabilities.

Outputs:
    A dictionary of rounded metric values.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_advanced_metrics(y_true, y_pred, y_proba=None) -> dict[str, float]:
    """Compute accuracy, precision, recall, F1, MCC, specificity, and optional AUC metrics."""
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall_sensitivity": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "mcc": round(float(matthews_corrcoef(y_true, y_pred)), 4),
    }

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    true_negative = int(((y_true_arr == 0) & (y_pred_arr == 0)).sum())
    false_positive = int(((y_true_arr == 0) & (y_pred_arr == 1)).sum())
    metrics["specificity"] = round(true_negative / max(true_negative + false_positive, 1), 4)

    if y_proba is not None:
        proba = np.asarray(y_proba)
        positive_score = proba[:, 1] if proba.ndim == 2 else proba
        metrics["roc_auc"] = round(float(roc_auc_score(y_true, positive_score)), 4)
        metrics["pr_auc"] = round(float(average_precision_score(y_true, positive_score)), 4)
        metrics["log_loss"] = round(float(log_loss(y_true, proba)), 4)

    return metrics
