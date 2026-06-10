"""Video file discovery and quality inspection helpers.

Purpose:
    Provide shared utilities for locating video files, hashing files, reading
    metadata, and assigning simple quality-review statuses.

Inputs:
    Local folders and video file paths.

Outputs:
    Lists of video paths, SHA-256 hashes, metadata dictionaries, and quality
    status labels.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2

from config.project_config import VIDEO_EXTENSIONS


def iter_video_files(folder: Path) -> list[Path]:
    """Return all supported video files below a folder in deterministic order."""
    if not folder.exists():
        return []
    return sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate a SHA-256 hash for a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_video(path: Path) -> dict[str, object]:
    """Read basic OpenCV metadata for a video file."""
    capture = cv2.VideoCapture(str(path))
    opened = capture.isOpened()
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) if opened else 0.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if opened else 0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) if opened else 0
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) if opened else 0
    duration = frame_count / fps if fps > 0 else 0.0
    capture.release()

    return {
        "opened": opened,
        "fps": round(fps, 2),
        "frame_count": frame_count,
        "duration": round(duration, 2),
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}" if width and height else "unknown",
    }


def video_quality_status(info: dict[str, object]) -> tuple[str, list[str]]:
    """Classify a video's basic usability based on metadata checks."""
    reasons: list[str] = []
    if not info["opened"]:
        reasons.append("cannot_open")
    if float(info["duration"]) < 2:
        reasons.append("too_short_under_2s")
    if int(info["width"]) < 224 or int(info["height"]) < 224:
        reasons.append("low_resolution")
    if int(info["frame_count"]) <= 0:
        reasons.append("missing_frames")

    if "cannot_open" in reasons or "missing_frames" in reasons:
        return "remove_or_reextract", reasons
    if reasons:
        return "review", reasons
    return "worthy", reasons
