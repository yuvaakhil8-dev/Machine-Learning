"""K-Nearest Neighbors model factory.

Purpose:
    Build the distance-based KNN classifier baseline.
"""

from sklearn.neighbors import KNeighborsClassifier


def build_model() -> KNeighborsClassifier:
    """Return the configured KNN classifier."""
    return KNeighborsClassifier(n_neighbors=7)
