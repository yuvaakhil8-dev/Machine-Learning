"""Classical video feature extraction.

Purpose:
    Extract interpretable frame-level and temporal descriptors from a video.

Inputs:
    A video path and sampling interval.

Outputs:
    A dictionary of numeric video-level features used by downstream fusion and
    model training steps.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def extract_classical_video_features(video_path: Path, seconds_per_sample: float = 1.0) -> dict[str, float]:
    """Extract brightness, motion, optical-flow, blur, texture, and color features."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return {
            "video_frame_brightness": 0.0,
            "video_motion_intensity": 0.0,
            "video_optical_flow_magnitude": 0.0,
            "video_sudden_movement_score": 0.0,
            "video_activity_acceleration": 0.0,
            "video_motion_variance": 0.0,
            "video_scene_transition_frequency": 0.0,
            "video_frame_difference_energy": 0.0,
            "video_blur_score": 0.0,
            "video_red_dominance": 0.0,
            "video_frame_texture_score": 0.0,
            "video_color_histogram_variation": 0.0,
            "video_temporal_motion_consistency": 0.0,
            "video_activity_intensity": 0.0,
            "video_dynamic_activity_score": 0.0,
        }

    fps = capture.get(cv2.CAP_PROP_FPS) or 1.0
    interval = max(int(round(fps * seconds_per_sample)), 1)
    frame_idx = 0
    brightness: list[float] = []
    blur_scores: list[float] = []
    red_ratios: list[float] = []
    motions: list[float] = []
    optical_flows: list[float] = []
    texture_scores: list[float] = []
    color_histograms: list[np.ndarray] = []
    transitions = 0
    previous_gray = None

    while True:
        frame_was_read, frame = capture.read()
        if not frame_was_read:
            break
        if frame_idx % interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness.append(float(np.mean(gray)))
            laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            blur_scores.append(laplacian_var)
            texture_scores.append(laplacian_var)
            blue_channel, green_channel, red_channel = cv2.split(frame)
            red_ratios.append(
                float(
                    np.mean(red_channel)
                    / max(np.mean(blue_channel) + np.mean(green_channel) + np.mean(red_channel), 1)
                )
            )
            color_histogram = cv2.calcHist(
                [frame],
                [0, 1, 2],
                None,
                [8, 8, 8],
                [0, 256, 0, 256, 0, 256],
            )
            color_histogram = cv2.normalize(color_histogram, color_histogram).flatten()
            color_histograms.append(color_histogram)
            if previous_gray is not None:
                motion = float(np.mean(cv2.absdiff(previous_gray, gray)))
                motions.append(motion)
                flow = cv2.calcOpticalFlowFarneback(
                    previous_gray,
                    gray,
                    None,
                    0.5,
                    2,
                    15,
                    2,
                    5,
                    1.1,
                    0,
                )
                magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                optical_flows.append(float(np.mean(magnitude)))
                if motion > 35:
                    transitions += 1
            previous_gray = gray
        frame_idx += 1

    capture.release()
    sampled = max(len(brightness), 1)
    mean_motion = float(np.mean(motions)) if motions else 0.0
    motion_acceleration = float(np.mean(np.abs(np.diff(motions)))) if len(motions) > 1 else 0.0
    motion_variance = float(np.var(motions)) if motions else 0.0
    sudden_movement = float(np.max(motions) - np.mean(motions)) if motions else 0.0
    motion_consistency = 1.0 / (1.0 + float(np.std(motions))) if motions else 1.0
    color_variation = float(np.mean(np.std(np.stack(color_histograms), axis=0))) if len(color_histograms) > 1 else 0.0
    optical_flow = float(np.mean(optical_flows)) if optical_flows else 0.0
    dynamic_activity = mean_motion + optical_flow + motion_acceleration
    return {
        "video_frame_brightness": float(np.mean(brightness)) if brightness else 0.0,
        "video_motion_intensity": mean_motion,
        "video_optical_flow_magnitude": optical_flow,
        "video_sudden_movement_score": sudden_movement,
        "video_activity_acceleration": motion_acceleration,
        "video_motion_variance": motion_variance,
        "video_scene_transition_frequency": transitions / sampled,
        "video_frame_difference_energy": float(np.mean(np.square(motions))) if motions else 0.0,
        "video_blur_score": float(np.mean(blur_scores)) if blur_scores else 0.0,
        "video_red_dominance": float(np.mean(red_ratios)) if red_ratios else 0.0,
        "video_frame_texture_score": float(np.mean(texture_scores)) if texture_scores else 0.0,
        "video_color_histogram_variation": color_variation,
        "video_temporal_motion_consistency": motion_consistency,
        "video_activity_intensity": mean_motion * (transitions + 1),
        "video_dynamic_activity_score": dynamic_activity,
    }
