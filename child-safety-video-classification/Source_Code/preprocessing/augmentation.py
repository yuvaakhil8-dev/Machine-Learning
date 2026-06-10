"""Frame and sequence augmentation utilities.

Purpose:
    Provide lightweight visual augmentation for sampled video frames.

Inputs:
    Normalized RGB frames in NumPy arrays.

Outputs:
    Augmented frames or frame sequences with the same shape.
"""

from __future__ import annotations

import cv2
import numpy as np


def augment_frame(frame: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Apply random flip, brightness, contrast, crop, rotation, noise, and blur transforms."""
    rng = rng or np.random.default_rng()
    augmented = frame.copy()

    if rng.random() < 0.5:
        augmented = np.flip(augmented, axis=1)
    if rng.random() < 0.35:
        factor = rng.uniform(0.85, 1.15)
        augmented = np.clip(augmented * factor, 0.0, 1.0)
    if rng.random() < 0.30:
        mean = augmented.mean(axis=(0, 1), keepdims=True)
        contrast = rng.uniform(0.85, 1.20)
        augmented = np.clip((augmented - mean) * contrast + mean, 0.0, 1.0)
    if rng.random() < 0.20:
        height, width = augmented.shape[:2]
        crop_scale = rng.uniform(0.88, 1.0)
        crop_h = max(1, int(height * crop_scale))
        crop_w = max(1, int(width * crop_scale))
        y0 = int(rng.integers(0, height - crop_h + 1))
        x0 = int(rng.integers(0, width - crop_w + 1))
        crop = augmented[y0:y0 + crop_h, x0:x0 + crop_w]
        augmented = cv2.resize(crop, (width, height))
    if rng.random() < 0.18:
        height, width = augmented.shape[:2]
        angle = float(rng.uniform(-8, 8))
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        augmented = cv2.warpAffine(augmented, matrix, (width, height), borderMode=cv2.BORDER_REFLECT_101)
    if rng.random() < 0.25:
        noise = rng.normal(0, 0.025, augmented.shape)
        augmented = np.clip(augmented + noise, 0.0, 1.0)
    if rng.random() < 0.20:
        augmented = cv2.GaussianBlur(augmented, (3, 3), 0)

    return augmented.astype(np.float32)


def augment_sequence(frames: np.ndarray, seed: int | None = None) -> np.ndarray:
    """Apply frame-level augmentation and optional temporal shift to a sequence."""
    rng = np.random.default_rng(seed)
    augmented = np.stack([augment_frame(frame, rng) for frame in frames]).astype(np.float32)
    if len(augmented) > 2 and rng.random() < 0.25:
        shift = int(rng.integers(1, min(4, len(augmented))))
        augmented = np.roll(augmented, shift=shift, axis=0)
    return augmented
