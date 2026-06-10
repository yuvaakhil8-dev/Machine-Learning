"""Face and expression proxy feature extraction.

Purpose:
    Build lightweight expression-related proxy features without requiring a
    heavier emotion-recognition dependency at runtime.

Inputs:
    RGB video frames normalized to the [0, 1] range.

Outputs:
    Numeric face-count, contrast, instability, and confidence proxy features.
"""

from __future__ import annotations

import cv2
import numpy as np


def extract_emotion_proxy_features(frames: np.ndarray) -> dict[str, float]:
    """Lightweight emotion proxy without requiring DeepFace/FER at runtime.

    For a small college dataset, this provides stable face/activity indicators while
    keeping DeepFace or FER as an easy future upgrade.
    """
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    face_counts: list[int] = []
    high_contrast_faces = 0
    contrast_values: list[float] = []

    for frame in frames:
        image = (frame * 255).astype(np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.1, 4)
        face_counts.append(len(faces))
        for face_x, face_y, face_width, face_height in faces:
            face_region = gray[face_y:face_y + face_height, face_x:face_x + face_width]
            if face_region.size:
                contrast = float(np.std(face_region))
                contrast_values.append(contrast)
                if contrast > 55:
                    high_contrast_faces += 1

    total_frames = max(len(frames), 1)
    angry_proxy = high_contrast_faces / total_frames
    instability = float(np.std(face_counts)) if face_counts else 0.0
    face_presence_ratio = float(np.mean([count > 0 for count in face_counts])) if face_counts else 0.0
    fear_proxy = min(1.0, (float(np.mean(contrast_values)) / 100.0) if contrast_values else 0.0)
    stress_proxy = min(1.0, (angry_proxy + instability) / 2.0)
    aggressive_expression = min(1.0, angry_proxy + stress_proxy)
    temporal_consistency = 1.0 / (1.0 + instability)
    emotion_confidence = min(1.0, 0.35 + 0.65 * face_presence_ratio) if face_counts else 0.0
    aggression_validation = min(1.0, aggressive_expression * emotion_confidence)
    return {
        "emotion_face_count_mean": float(np.mean(face_counts)) if face_counts else 0.0,
        "emotion_face_count_max": float(np.max(face_counts)) if face_counts else 0.0,
        "emotion_angry_face_ratio_proxy": angry_proxy,
        "emotion_fear_score_proxy": fear_proxy,
        "emotion_stress_score_proxy": stress_proxy,
        "emotion_aggressive_expression_score_proxy": aggressive_expression,
        "emotion_instability_proxy": instability,
        "emotion_temporal_consistency_score": temporal_consistency,
        "emotion_multiframe_aggregation_score": float(np.mean(contrast_values)) if contrast_values else 0.0,
        "emotion_confidence_score": emotion_confidence,
        "emotion_aggression_validation_score": aggression_validation,
        "emotion_temporal_behavior_validation": min(1.0, temporal_consistency * emotion_confidence),
        "emotion_confidence_calibrated_score": min(1.0, (0.6 * aggression_validation) + (0.4 * stress_proxy)),
    }
