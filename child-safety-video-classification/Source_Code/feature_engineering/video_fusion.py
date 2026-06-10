"""Feature-fusion utilities for video-level safety classification.

Purpose:
    Combine motion, object, expression, pose, and quality descriptors into
    derived proxy scores used by the downstream classifiers.

Inputs:
    A dictionary containing extracted numeric feature values.

Outputs:
    The same dictionary enriched with fusion feature keys.
"""

from __future__ import annotations


def add_video_fusion_features(row: dict[str, float]) -> dict[str, float]:
    """Create video-only risk proxy features from visual descriptors."""
    motion = float(row.get("video_motion_intensity", 0.0))
    activity = float(row.get("video_activity_intensity", 0.0))
    transitions = float(row.get("video_scene_transition_frequency", 0.0))
    blur = float(row.get("video_blur_score", 0.0))
    weapon = float(row.get("yolo_weapon_detection_probability", row.get("yolo_unsafe_object_probability", 0.0)))
    emotion = float(
        row.get(
            "emotion_aggression_validation_score",
            row.get("emotion_aggressive_expression_score_proxy", row.get("emotion_angry_face_ratio_proxy", 0.0)),
        )
    )
    emotion_consistency = float(row.get("emotion_temporal_consistency_score", 1.0))
    pose = float(row.get("pose_aggressive_movement_score", row.get("pose_instability_proxy", 0.0)))
    pose_consistency = float(row.get("temporal_pose_consistency_proxy", 1.0))
    suspicious_activity = float(row.get("pose_suspicious_activity_probability", pose))
    object_confidence = float(row.get("yolo_object_detection_confidence", weapon))
    motion_consistency = float(row.get("video_temporal_motion_consistency", 1.0))

    row["fusion_video_activity_score"] = motion + activity
    row["fusion_temporal_change_score"] = motion * (1.0 + transitions)
    row["fusion_visual_quality_score"] = blur
    row["fusion_final_video_confidence_proxy"] = min(1.0, (motion + activity + transitions) / 100.0)
    row["fusion_unsafe_activity_score"] = min(1.0, weapon + emotion + pose + transitions)
    row["fusion_violence_probability_proxy"] = min(1.0, (weapon * 0.4) + (pose * 0.35) + (emotion * 0.25))
    row["fusion_aggression_confidence_score"] = min(1.0, (0.45 * emotion) + (0.35 * pose) + (0.20 * suspicious_activity))
    row["fusion_temporal_behavior_consistency_score"] = min(1.0, (0.45 * motion_consistency) + (0.30 * pose_consistency) + (0.25 * emotion_consistency))
    row["fusion_child_safety_risk_score"] = min(
        1.0,
        (0.30 * row["fusion_unsafe_activity_score"])
        + (0.25 * row["fusion_aggression_confidence_score"])
        + (0.20 * weapon)
        + (0.15 * object_confidence)
        + (0.10 * (1.0 - row["fusion_temporal_behavior_consistency_score"])),
    )
    row["fusion_final_multimodal_feature_score"] = (
        row["fusion_video_activity_score"]
        + row["fusion_temporal_change_score"]
        + row["fusion_unsafe_activity_score"]
        + row["fusion_violence_probability_proxy"]
    )
    return row
