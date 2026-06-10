from __future__ import annotations


def cnn_feature_name(index: int) -> str:
    if index < 128:
        return f"edge_texture_feature_{index + 1:03d}"
    if index < 256:
        return f"object_body_feature_{index - 127:03d}"
    if index < 384:
        return f"scene_activity_feature_{index - 255:03d}"
    return f"safety_context_feature_{index - 383:03d}"


def is_cnn_feature(column: str) -> bool:
    return column.startswith(
        (
            "edge_texture_feature_",
            "object_body_feature_",
            "scene_activity_feature_",
            "safety_context_feature_",
        )
    )


def legacy_visual_to_cnn_name(column: str) -> str:
    if column.startswith("visual_"):
        return cnn_feature_name(int(column.split("_")[1]))
    return column

