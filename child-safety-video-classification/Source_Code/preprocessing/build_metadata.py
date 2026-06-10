"""Dataset metadata builder.

Purpose:
    Scan configured dataset folders, inspect each video, assign labels, and
    write a metadata CSV for downstream preprocessing and auditing.

Inputs:
    Configured safe/unsafe dataset folders.

Outputs:
    `dataset_metadata.csv` or a caller-provided output path.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from config.project_config import DATASET_DIRS, LABELS, METADATA_PATH, PROJECT_ROOT
from utils.video_io import file_sha256, inspect_video, iter_video_files, video_quality_status


def load_online_source_map() -> dict[str, str]:
    """Load optional dataset-source manifests if present."""
    source_map: dict[str, str] = {}
    for manifest in (PROJECT_ROOT / "dataset_sources").glob("*.csv"):
        df = pd.read_csv(manifest)
        if "target_path" not in df.columns:
            continue
        for _, row in df.iterrows():
            source_map[str(row["target_path"]).replace("/", "\\")] = str(row.get("source_repository", manifest.name))
    return source_map


def infer_source(relative_path: str, source_map: dict[str, str]) -> str:
    """Resolve a video source label from a manifest map or fall back to local data."""
    normalized = relative_path.replace("/", "\\")
    if normalized in source_map:
        return source_map[normalized]
    return "local_original_dataset"


def build_metadata(output_path: Path = METADATA_PATH) -> pd.DataFrame:
    """Build and save video metadata rows for every configured class folder."""
    source_map = load_online_source_map()
    rows: list[dict[str, object]] = []

    for class_name, folder in DATASET_DIRS.items():
        for video_path in iter_video_files(folder):
            relative_path = str(video_path.relative_to(PROJECT_ROOT))
            info = inspect_video(video_path)
            status, reasons = video_quality_status(info)
            rows.append(
                {
                    "video_path": relative_path,
                    "label": LABELS[class_name],
                    "class_name": class_name,
                    "source": infer_source(relative_path, source_map),
                    "duration": info["duration"],
                    "resolution": info["resolution"],
                    "language": "unknown",
                    "fps": info["fps"],
                    "frame_count": info["frame_count"],
                    "file_size_mb": round(video_path.stat().st_size / (1024 * 1024), 2),
                    "sha256": file_sha256(video_path),
                    "quality_status": status,
                    "quality_notes": ";".join(reasons),
                }
            )

    metadata = pd.DataFrame(rows)
    metadata.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)
    return metadata


def main() -> None:
    """Command-line entry point for metadata generation."""
    metadata = build_metadata()
    print(f"Saved {len(metadata)} metadata rows to {METADATA_PATH}")
    print(metadata["class_name"].value_counts().to_string())
    print(metadata["quality_status"].value_counts().to_string())


if __name__ == "__main__":
    main()
