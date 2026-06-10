"""Random Forest model factory.

Purpose:
    Build the Random Forest classifier used for nonlinear ensemble baselines.
"""

from sklearn.ensemble import RandomForestClassifier


def build_model(random_state: int = 42) -> RandomForestClassifier:
    """Return the configured Random Forest classifier."""
    return RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
