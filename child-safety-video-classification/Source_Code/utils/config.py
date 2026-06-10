from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".wmv"}

DATASET_DIRS = {
    "safe": PROJECT_ROOT / "Safe (3)",
    "unsafe": PROJECT_ROOT / "Unsafe",
}

LABELS = {
    "safe": 0,
    "unsafe": 1,
}

STANDARD_DATASET_DIR = PROJECT_ROOT / "dataset"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
METADATA_PATH = PROJECT_ROOT / "dataset_metadata.csv"
