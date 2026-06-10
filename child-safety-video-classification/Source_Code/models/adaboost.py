"""AdaBoost model factory.

Purpose:
    Build the AdaBoost classifier used as a boosting baseline.
"""

from sklearn.ensemble import AdaBoostClassifier


def build_model(random_state: int = 42) -> AdaBoostClassifier:
    """Return the configured AdaBoost classifier."""
    return AdaBoostClassifier(n_estimators=150, learning_rate=0.5, random_state=random_state)
