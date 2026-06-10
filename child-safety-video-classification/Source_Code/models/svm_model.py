"""Support Vector Machine model factory.

Purpose:
    Build the SVM classifier used for high-dimensional video feature vectors.
"""

from sklearn.svm import SVC


def build_model(random_state: int = 42) -> SVC:
    """Return the configured RBF-kernel SVM classifier."""
    return SVC(
        kernel="rbf",
        C=2.0,
        gamma="scale",
        probability=True,
        class_weight="balanced",
        random_state=random_state,
    )
