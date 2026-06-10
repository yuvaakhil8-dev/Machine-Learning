from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EXCLUDED_COLUMNS = {"video_name", "video_path", "class_name", "label"}


@dataclass(frozen=True)
class FixedFeatureVectorInfo:
    feature_count: int
    feature_names: list[str]


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return trainable numeric columns while excluding identifiers and label."""
    return [
        column
        for column in df.columns
        if column not in EXCLUDED_COLUMNS and pd.api.types.is_numeric_dtype(df[column])
    ]


def build_zscore_normalization_pipeline() -> Pipeline:
    """Impute missing values and apply z-score standardization.

    z = (x - mean) / standard_deviation
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
        ]
    )


def fit_transform_features(df: pd.DataFrame) -> tuple[np.ndarray, Pipeline, FixedFeatureVectorInfo]:
    """Create a fixed-length standardized feature matrix from a feature CSV."""
    columns = numeric_feature_columns(df)
    pipeline = build_zscore_normalization_pipeline()
    matrix = pipeline.fit_transform(df[columns])
    return matrix, pipeline, FixedFeatureVectorInfo(len(columns), columns)


def validate_fixed_length(df: pd.DataFrame, expected_columns: list[str]) -> dict[str, object]:
    """Check whether every expected feature column exists for model input."""
    missing = [column for column in expected_columns if column not in df.columns]
    extra = [column for column in df.columns if column not in set(expected_columns) | EXCLUDED_COLUMNS]
    return {
        "is_valid": not missing,
        "expected_feature_count": len(expected_columns),
        "missing_features": missing,
        "extra_numeric_features": extra,
    }
