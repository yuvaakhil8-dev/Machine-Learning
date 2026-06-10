from __future__ import annotations

import pandas as pd


def build_error_analysis_table(metadata: pd.DataFrame, y_true, y_pred, y_proba=None) -> pd.DataFrame:
    result = metadata.copy().reset_index(drop=True)
    result["true_label"] = list(y_true)
    result["predicted_label"] = list(y_pred)
    result["is_error"] = result["true_label"] != result["predicted_label"]
    result["error_type"] = "correct"
    result.loc[(result["true_label"] == 0) & (result["predicted_label"] == 1), "error_type"] = "false_positive"
    result.loc[(result["true_label"] == 1) & (result["predicted_label"] == 0), "error_type"] = "false_negative"
    if y_proba is not None:
        result["unsafe_confidence"] = y_proba[:, 1]
        result["confidence_margin"] = (y_proba.max(axis=1) - y_proba.min(axis=1))
    return result

