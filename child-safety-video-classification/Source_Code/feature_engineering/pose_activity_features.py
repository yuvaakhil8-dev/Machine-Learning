"""Pose and activity proxy feature extraction.

Purpose:
    Estimate activity and posture-change proxies from frame differences without
    requiring heavyweight pose-estimation runtimes.

Inputs:
    RGB frame arrays normalized to the [0, 1] range.

Outputs:
    Numeric proxy features for pose instability, velocity, abnormal activity,
    and human-interaction intensity.
"""

from __future__ import annotations

import numpy as np


def extract_pose_proxy_features(frames: np.ndarray) -> dict[str, float]:
    """Pose/activity proxies based on frame-to-frame visual change.

    MediaPipe/OpenPose can replace this module later. These proxies stay useful on
    small datasets where installing pose stacks may be heavy.
    """
    if len(frames) < 2:
        return {
            "pose_instability_proxy": 0.0,
            "pose_velocity_proxy": 0.0,
            "hand_leg_motion_proxy": 0.0,
            "temporal_pose_consistency_proxy": 1.0,
            "pose_abnormal_activity_score": 0.0,
            "pose_human_interaction_intensity": 0.0,
        }

    diffs = np.abs(np.diff(frames, axis=0))
    motion_per_frame = diffs.mean(axis=(1, 2, 3))
    upper_motion = diffs[:, : frames.shape[1] // 2, :, :].mean()
    lower_motion = diffs[:, frames.shape[1] // 2 :, :, :].mean()
    velocity = float(np.mean(motion_per_frame))
    instability = float(np.std(motion_per_frame))
    aggressive_score = min(1.0, velocity * 10.0 + instability * 10.0)
    temporal_consistency = 1.0 / (1.0 + instability)
    body_region_imbalance = float(abs(upper_motion - lower_motion))
    abnormal_activity = min(1.0, (0.55 * aggressive_score) + (body_region_imbalance * 8.0) + ((1.0 - temporal_consistency) * 0.25))

    return {
        "pose_instability_proxy": instability,
        "pose_velocity_proxy": velocity,
        "hand_leg_motion_proxy": float(upper_motion + lower_motion),
        "temporal_pose_consistency_proxy": temporal_consistency,
        "pose_movement_intensity": velocity,
        "pose_aggressive_movement_score": aggressive_score,
        "pose_suspicious_activity_probability": aggressive_score,
        "pose_abnormal_activity_score": abnormal_activity,
        "pose_human_interaction_intensity": min(1.0, velocity * 8.0 + body_region_imbalance * 4.0),
    }
