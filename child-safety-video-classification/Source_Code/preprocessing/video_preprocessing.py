"""Video validation and frame preprocessing utilities.

Purpose:
    Provide reusable helpers for checking whether a video can be read and for
    converting video frames into fixed-size normalized arrays.

Inputs:
    Local video files readable by OpenCV.

Outputs:
    Boolean validation results and NumPy arrays shaped for downstream feature
    extraction.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def validate_video(video_path: Path) -> bool:
    """Return True when OpenCV can read at least one frame from a video."""
    capture = cv2.VideoCapture(str(video_path))
    is_valid = capture.isOpened() and int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0) > 0
    capture.release()
    return is_valid


def sample_frames(
    video_path: Path,
    *,
    max_frames: int = 16,
    size: tuple[int, int] = (224, 224),
    every_n_frames: int = 5,
) -> np.ndarray:
    """Sample, resize, RGB-convert, and normalize frames from a video.

    If the video has fewer frames than requested, the last available frame is
    repeated so downstream models always receive a fixed-length sequence.
    """
    capture = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    frame_index = 0

    while capture.isOpened() and len(frames) < max_frames:
        frame_was_read, frame = capture.read()
        if not frame_was_read:
            break
        if frame_index % every_n_frames == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, size)
            frames.append(frame.astype(np.float32) / 255.0)
        frame_index += 1

    capture.release()

    if not frames:
        return np.zeros((max_frames, size[1], size[0], 3), dtype=np.float32)

    while len(frames) < max_frames:
        frames.append(frames[-1].copy())

    return np.stack(frames[:max_frames]).astype(np.float32)


def remove_duplicate_frames(frames: np.ndarray, threshold: float = 0.995) -> np.ndarray:
    """Remove consecutive near-duplicate frames using mean absolute difference."""
    if len(frames) <= 1:
        return frames
    unique = [frames[0]]
    for frame in frames[1:]:
        similarity = 1.0 - float(np.mean(np.abs(unique[-1] - frame)))
        if similarity < threshold:
            unique.append(frame)
    return np.stack(unique).astype(np.float32)
