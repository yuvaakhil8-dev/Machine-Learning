"""Dataset preparation entry point.

Purpose:
    Build metadata, summarize duplicates and quality status, and save a dataset
    preparation audit report.

Outputs:
    A metadata DataFrame and `dataset_audit/dataset_preparation_summary.json`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.project_config import METADATA_PATH, PROJECT_ROOT
from preprocessing.build_metadata import build_metadata


def prepare_dataset() -> pd.DataFrame:
    """Build metadata and write a compact dataset preparation summary."""
    metadata = build_metadata()
    duplicate_count = int(metadata.duplicated("sha256").sum())
    balanced_counts = metadata.groupby("class_name").size().to_dict()
    report = {
        "total_videos": int(len(metadata)),
        "class_counts": balanced_counts,
        "duplicate_hash_count": duplicate_count,
        "quality_status": metadata["quality_status"].value_counts().to_dict(),
    }

    report_path = PROJECT_ROOT / "dataset_audit" / "dataset_preparation_summary.json"
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(pd.Series(report).to_json(indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    """Command-line entry point for dataset preparation."""
    metadata = prepare_dataset()
    print(f"Prepared dataset metadata: {METADATA_PATH}")
    print(metadata.groupby(["class_name", "quality_status"]).size().to_string())


if __name__ == "__main__":
    main()
