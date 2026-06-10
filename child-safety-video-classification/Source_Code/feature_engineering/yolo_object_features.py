"""YOLO object and person feature extraction.

Purpose:
    Use a YOLO-compatible detector to summarize object, person, and selected
    safety-relevant object cues at video level.

Inputs:
    A video path and an initialized YOLO model.

Outputs:
    Numeric YOLO-derived features for downstream fusion and model training.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


COCO_DANGEROUS_NAMES = {"knife", "scissors"}
SUSPICIOUS_OBJECT_NAMES = {"knife", "scissors", "baseball bat"}


def extract_yolo_features(
    video_path: Path,
    model,
    *,
    seconds_per_sample: float = 1.0,
) -> dict[str, float]:
    """Sample video frames and aggregate YOLO detection counts and confidences."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return {
            "yolo_person_count_mean": 0.0,
            "yolo_person_count_max": 0.0,
            "yolo_crowd_density": 0.0,
            "yolo_object_count_mean": 0.0,
            "yolo_dangerous_object_count": 0.0,
            "yolo_unsafe_object_probability": 0.0,
            "yolo_weapon_detection_probability": 0.0,
            "yolo_suspicious_object_count": 0.0,
            "yolo_object_detection_confidence": 0.0,
            "yolo_human_interaction_intensity": 0.0,
        }

    fps = capture.get(cv2.CAP_PROP_FPS) or 1.0
    interval = max(int(round(fps * seconds_per_sample)), 1)
    frame_index = 0
    person_counts: list[int] = []
    object_counts: list[int] = []
    dangerous_hits = 0
    suspicious_hits = 0
    confidence_values: list[float] = []
    sampled = 0

    names = getattr(model, "names", {})

    while True:
        frame_was_read, frame = capture.read()
        if not frame_was_read:
            break
        if frame_index % interval == 0:
            sampled += 1
            result_list = model(frame, verbose=False)
            frame_persons = 0
            frame_objects = 0
            for result in result_list:
                boxes = result.boxes
                frame_objects += len(boxes)
                if getattr(boxes, "conf", None) is not None:
                    confidence_values.extend([float(conf) for conf in boxes.conf])
                for cls in boxes.cls:
                    class_id = int(cls)
                    class_name = str(names.get(class_id, class_id)).lower()
                    if class_id == 0:
                        frame_persons += 1
                    if class_name in COCO_DANGEROUS_NAMES:
                        dangerous_hits += 1
                    if class_name in SUSPICIOUS_OBJECT_NAMES:
                        suspicious_hits += 1
            person_counts.append(frame_persons)
            object_counts.append(frame_objects)
        frame_index += 1

    capture.release()
    sampled = max(sampled, 1)
    return {
        "yolo_person_count_mean": float(np.mean(person_counts)) if person_counts else 0.0,
        "yolo_person_count_max": float(np.max(person_counts)) if person_counts else 0.0,
        "yolo_crowd_density": float(np.mean([count >= 5 for count in person_counts])) if person_counts else 0.0,
        "yolo_object_count_mean": float(np.mean(object_counts)) if object_counts else 0.0,
        "yolo_dangerous_object_count": float(dangerous_hits),
        "yolo_unsafe_object_probability": min(1.0, dangerous_hits / sampled),
        "yolo_weapon_detection_probability": min(1.0, dangerous_hits / sampled),
        "yolo_suspicious_object_count": float(suspicious_hits),
        "yolo_object_detection_confidence": float(np.mean(confidence_values)) if confidence_values else 0.0,
        "yolo_human_interaction_intensity": float(np.mean(person_counts) * np.mean(object_counts)) if person_counts and object_counts else 0.0,
    }
