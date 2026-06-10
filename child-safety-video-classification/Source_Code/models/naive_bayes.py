"""Naive Bayes model factory.

Purpose:
    Build the Gaussian Naive Bayes probabilistic baseline.
"""

from sklearn.naive_bayes import GaussianNB


def build_model() -> GaussianNB:
    """Return the configured Gaussian Naive Bayes classifier."""
    return GaussianNB()
