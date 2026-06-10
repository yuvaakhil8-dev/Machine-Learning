"""Logistic Regression model factory.

Purpose:
    Build the Logistic Regression classifier used as a linear baseline.

Inputs:
    Random seed for reproducible initialization.

Outputs:
    An initialized scikit-learn LogisticRegression estimator.
"""

from sklearn.linear_model import LogisticRegression


def build_model(random_state: int = 42) -> LogisticRegression:
    """Return the configured Logistic Regression classifier."""
    return LogisticRegression(max_iter=3000, random_state=random_state)
