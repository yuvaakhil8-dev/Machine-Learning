"""Feature standardization helpers.

Purpose:
    Build a reusable imputation and z-score scaling pipeline for numeric video
    features.
"""

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_standardization_pipeline() -> Pipeline:
    """Return a pipeline that fills missing values with zero and standardizes features."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
        ]
    )
