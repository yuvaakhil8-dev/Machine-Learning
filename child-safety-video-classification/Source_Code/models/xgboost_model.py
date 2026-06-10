"""XGBoost model factory.

Purpose:
    Build the XGBoost classifier used for boosted-tree experiments.
"""

from xgboost import XGBClassifier


def build_model(random_state: int = 42) -> XGBClassifier:
    """Return the configured XGBoost classifier."""
    return XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=random_state,
    )
