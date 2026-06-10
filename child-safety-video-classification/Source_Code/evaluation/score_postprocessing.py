from __future__ import annotations

import numpy as np


def temporal_smooth_scores(scores: list[float] | np.ndarray, window_size: int = 3) -> np.ndarray:
    """Smooth frame-level scores with a moving average.

    This is useful when frame-level predictions fluctuate because of blur,
    temporary occlusion, or sudden camera motion.
    """
    values = np.asarray(scores, dtype=float)
    if values.size == 0:
        return values
    window_size = max(1, int(window_size))
    if window_size == 1 or values.size < 2:
        return values
    kernel = np.ones(window_size, dtype=float) / window_size
    padded = np.pad(values, (window_size // 2, window_size - 1 - window_size // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def aggregate_video_score(frame_scores: list[float] | np.ndarray, method: str = "mean") -> float:
    """Convert frame-level scores into one video-level score."""
    values = np.asarray(frame_scores, dtype=float)
    if values.size == 0:
        return 0.5
    method = method.lower()
    if method == "max":
        return float(np.max(values))
    if method == "median":
        return float(np.median(values))
    return float(np.mean(values))


def calibrate_confidence(probability: float, temperature: float = 1.0) -> float:
    """Lightweight probability calibration using temperature scaling.

    temperature > 1 softens overconfident probabilities.
    temperature < 1 sharpens underconfident probabilities.
    """
    probability = float(np.clip(probability, 1e-6, 1.0 - 1e-6))
    temperature = max(float(temperature), 1e-6)
    logit = np.log(probability / (1.0 - probability))
    calibrated = 1.0 / (1.0 + np.exp(-(logit / temperature)))
    return float(np.clip(calibrated, 0.0, 1.0))


def safe_unsafe_label_from_score(score: float, threshold: float = 0.5) -> str:
    return "Unsafe" if float(score) >= threshold else "Safe"
