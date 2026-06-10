"""Feature consistency validation utilities.

Purpose:
    Summarize multimodal feature groups and verify that a processed feature CSV
    contains labels and numeric trainable columns.

Inputs:
    A pandas DataFrame containing processed video features.

Outputs:
    Summary DataFrames and dictionaries for audit/report generation.
"""

from __future__ import annotations

import pandas as pd


FEATURE_PREFIX_GROUPS = {
    "feature_": "cnn_embedding",
    "video_": "video_motion_quality",
    "yolo_": "object_detection",
    "emotion_": "expression_proxy",
    "pose_": "activity_proxy",
    "hand_": "activity_proxy",
    "temporal_pose_": "activity_proxy",
    "fusion_": "feature_fusion",
}


def feature_group_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize feature counts and missing values by multimodal group."""
    rows: list[dict[str, object]] = []
    for prefix, group in FEATURE_PREFIX_GROUPS.items():
        columns = [column for column in df.columns if column.startswith(prefix)]
        if not columns:
            continue
        values = df[columns]
        rows.append(
            {
                "feature_group": group,
                "prefix": prefix,
                "feature_count": len(columns),
                "missing_values": int(values.isna().sum().sum()),
                "zero_values": int((values == 0).sum().sum()),
                "all_columns_present": True,
            }
        )
    return pd.DataFrame(rows)


def validate_multimodal_consistency(df: pd.DataFrame) -> dict[str, object]:
    """Validate fixed rows, labels, numeric features, and multimodal coverage."""
    numeric_features = [
        column
        for column in df.columns
        if column not in {"video_name", "video_path", "class_name", "label"}
        and pd.api.types.is_numeric_dtype(df[column])
    ]
    group_summary = feature_group_summary(df)
    return {
        "video_rows": int(len(df)),
        "numeric_feature_count": int(len(numeric_features)),
        "has_label_column": "label" in df.columns,
        "has_safe_and_unsafe_labels": sorted(df["label"].dropna().unique().tolist()) == [0, 1]
        if "label" in df.columns
        else False,
        "multimodal_group_count": int(len(group_summary)),
        "feature_groups": group_summary.to_dict(orient="records"),
    }
