"""MLP Classifier model factory.

Purpose:
    Build the feed-forward neural-network baseline over extracted features.
"""

from sklearn.neural_network import MLPClassifier


def build_model(random_state: int = 42) -> MLPClassifier:
    """Return the configured MLP classifier."""
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        alpha=0.001,
        early_stopping=True,
        validation_fraction=0.15,
        max_iter=600,
        random_state=random_state,
    )
