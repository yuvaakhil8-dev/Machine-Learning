from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import cv2
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".wmv"}
DATASET_DIRS = {
    "safe": PROJECT_ROOT / "Safe (3)",
    "unsafe": PROJECT_ROOT / "Unsafe",
}


def iter_videos(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def inspect_video(path: Path) -> dict[str, object]:
    cap = cv2.VideoCapture(str(path))
    opened = cap.isOpened()
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) if opened else 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if opened else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) if opened else 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) if opened else 0
    duration = frame_count / fps if fps > 0 else 0.0
    cap.release()

    reasons: list[str] = []
    if not opened:
        reasons.append("cannot_open")
    if duration < 2:
        reasons.append("too_short_under_2s")
    if duration > 180:
        reasons.append("very_long_over_3min")
    if width < 224 or height < 224:
        reasons.append("low_resolution")
    if fps <= 0:
        reasons.append("missing_fps")
    if frame_count <= 0:
        reasons.append("missing_frames")

    status = "worthy"
    if "cannot_open" in reasons or "missing_frames" in reasons:
        status = "remove_or_reextract"
    elif reasons:
        status = "review"

    return {
        "video_name": path.name,
        "relative_path": str(path.relative_to(PROJECT_ROOT)),
        "file_size_mb": round(path.stat().st_size / (1024 * 1024), 2),
        "opened": opened,
        "fps": round(fps, 2),
        "frame_count": frame_count,
        "duration_seconds": round(duration, 2),
        "width": width,
        "height": height,
        "status": status,
        "review_reasons": ";".join(reasons),
    }


def main() -> None:
    rows: list[dict[str, object]] = []
    for class_name, folder in DATASET_DIRS.items():
        for video_path in iter_videos(folder):
            row = {"class_name": class_name}
            row.update(inspect_video(video_path))
            rows.append(row)

    if not rows:
        raise SystemExit("No videos found in Safe (3) or Unsafe folders.")

    df = pd.DataFrame(rows)
    output_dir = PROJECT_ROOT / "outputs" / "audit" / "dataset_audit"
    output_dir.mkdir(exist_ok=True)
    df.to_csv(output_dir / "video_quality_audit.csv", index=False)

    summary = {
        "total_videos": int(len(df)),
        "class_counts": df["class_name"].value_counts().to_dict(),
        "status_counts": df["status"].value_counts().to_dict(),
        "avg_duration_by_class": df.groupby("class_name")["duration_seconds"].mean().round(2).to_dict(),
        "median_duration_by_class": df.groupby("class_name")["duration_seconds"].median().round(2).to_dict(),
        "min_duration_seconds": float(df["duration_seconds"].min()),
        "max_duration_seconds": float(df["duration_seconds"].max()),
        "low_resolution_count": int(df["review_reasons"].str.contains("low_resolution", na=False).sum()),
        "too_short_count": int(df["review_reasons"].str.contains("too_short", na=False).sum()),
        "cannot_open_count": int((~df["opened"]).sum()),
    }
    (output_dir / "video_quality_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved audit CSV to {output_dir / 'video_quality_audit.csv'}")


if __name__ == "__main__":
    main()
