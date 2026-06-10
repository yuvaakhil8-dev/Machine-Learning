from __future__ import annotations


def cnn_feature_name(index: int) -> str:
    """Return a presentation-friendly name for a CNN embedding dimension.

    CNN embedding dimensions are learned automatically, so they do not have exact
    human labels. These names group them by the visual abstraction level they
    mostly represent in a pretrained CNN.
    """
    if index < 128:
        return f"cnn_edge_texture_pattern_{index + 1:03d}"
    if index < 256:
        return f"cnn_object_body_part_pattern_{index - 127:03d}"
    if index < 384:
        return f"cnn_scene_activity_context_{index - 255:03d}"
    return f"cnn_high_level_safety_context_{index - 383:03d}"


def cnn_feature_group(name: str) -> str:
    if name.startswith("cnn_edge_texture_pattern_"):
        return "Low-level edges, texture, and contrast patterns"
    if name.startswith("cnn_object_body_part_pattern_"):
        return "Object, person, and body-part visual patterns"
    if name.startswith("cnn_scene_activity_context_"):
        return "Scene layout and activity-context patterns"
    if name.startswith("cnn_high_level_safety_context_"):
        return "High-level combined safety-context patterns"
    return "Video metadata"


def is_cnn_feature(name: str) -> bool:
    return name.startswith(
        (
            "cnn_edge_texture_pattern_",
            "cnn_object_body_part_pattern_",
            "cnn_scene_activity_context_",
            "cnn_high_level_safety_context_",
        )
    )


def cnn_feature_dictionary() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index in range(512):
        original = f"visual_{index:03d}"
        renamed = cnn_feature_name(index)
        rows.append(
            {
                "original_name": original,
                "presentation_name": renamed,
                "feature_group": cnn_feature_group(renamed),
                "meaning": "Learned CNN embedding dimension extracted from sampled video frames.",
            }
        )
    return rows
