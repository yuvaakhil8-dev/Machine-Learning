"""Decision Tree model factory.

Purpose:
    Build the Decision Tree classifier used as an interpretable tree baseline.
"""

from sklearn.tree import DecisionTreeClassifier


def build_model(random_state: int = 42) -> DecisionTreeClassifier:
    """Return the configured Decision Tree classifier."""
    return DecisionTreeClassifier(
        max_depth=8,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=random_state,
    )
