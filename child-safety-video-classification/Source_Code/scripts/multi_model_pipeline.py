from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.models import (
    MobileNet_V2_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    mobilenet_v2,
    mobilenet_v3_small,
    resnet18,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "Source_Code"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from feature_engineering.cnn_feature_names import cnn_feature_name
from feature_engineering.emotion_features import extract_emotion_proxy_features
from feature_engineering.pose_activity_features import extract_pose_proxy_features
from feature_engineering.video_features import extract_classical_video_features
from feature_engineering.video_fusion import add_video_fusion_features
from feature_engineering.yolo_object_features import extract_yolo_features


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".wmv"}

EXPLICIT_FEATURE_EXPLANATIONS = {
    "frames_sampled": (
        "Video metadata",
        "OpenCV video reader",
        "Number of frames sampled from the video for CNN feature extraction.",
    ),
    "video_fps": (
        "Video metadata",
        "OpenCV video reader",
        "Frames per second reported by the video file.",
    ),
    "video_duration_seconds": (
        "Video metadata",
        "OpenCV video reader",
        "Video duration estimated from total frames divided by FPS.",
    ),
    "video_frame_brightness": (
        "Classical visual features",
        "Grayscale frame statistics",
        "Average brightness of sampled frames.",
    ),
    "video_motion_intensity": (
        "Classical visual features",
        "Frame-to-frame grayscale difference",
        "Average visual change between sampled frames.",
    ),
    "video_optical_flow_magnitude": (
        "Motion analysis features",
        "Farneback optical flow",
        "Average optical-flow magnitude between sampled frames.",
    ),
    "video_activity_acceleration": (
        "Motion analysis features",
        "Motion difference derivative",
        "Average acceleration/change in motion intensity over time.",
    ),
    "video_scene_transition_frequency": (
        "Classical visual features",
        "Frame-to-frame grayscale difference",
        "Ratio of sampled frames with large scene changes.",
    ),
    "video_blur_score": (
        "Classical visual features",
        "Laplacian variance",
        "Average sharpness/blur score of sampled frames.",
    ),
    "video_red_dominance": (
        "Classical visual features",
        "RGB channel statistics",
        "Average red-channel dominance in sampled frames.",
    ),
    "video_frame_texture_score": (
        "Classical visual features",
        "Laplacian texture statistics",
        "Texture/detail strength in sampled frames.",
    ),
    "video_color_histogram_variation": (
        "Classical visual features",
        "RGB color histogram statistics",
        "Variation in color distribution across sampled frames.",
    ),
    "video_temporal_motion_consistency": (
        "Motion analysis features",
        "Frame-to-frame motion statistics",
        "Stability of motion patterns over time.",
    ),
    "video_activity_intensity": (
        "Classical visual features",
        "Motion and transition fusion",
        "Motion score boosted by scene-change activity.",
    ),
    "video_dynamic_activity_score": (
        "Motion analysis features",
        "Motion, optical flow, and acceleration fusion",
        "Dynamic activity score combining motion intensity, optical flow, and acceleration.",
    ),
    "yolo_person_count_mean": (
        "YOLO object/person features",
        "YOLOv8 object detector",
        "Average number of detected people per sampled frame.",
    ),
    "yolo_person_count_max": (
        "YOLO object/person features",
        "YOLOv8 object detector",
        "Maximum number of detected people in any sampled frame.",
    ),
    "yolo_crowd_density": (
        "YOLO object/person features",
        "YOLOv8 object detector",
        "Fraction of sampled frames with five or more detected people.",
    ),
    "yolo_object_count_mean": (
        "YOLO object/person features",
        "YOLOv8 object detector",
        "Average number of detected objects per sampled frame.",
    ),
    "yolo_dangerous_object_count": (
        "YOLO object/person features",
        "YOLOv8 object detector",
        "Count of detected COCO dangerous-object classes such as knife or scissors.",
    ),
    "yolo_unsafe_object_probability": (
        "YOLO object/person features",
        "YOLOv8 object detector",
        "Ratio-style proxy based on dangerous-object detections.",
    ),
    "yolo_weapon_detection_probability": (
        "YOLO object/person features",
        "YOLOv8 object detector",
        "Weapon/dangerous-object probability proxy based on detected dangerous classes.",
    ),
    "yolo_suspicious_object_count": (
        "YOLO object/person features",
        "YOLOv8 object detector",
        "Count of suspicious objects such as knife, scissors, or bat-like objects.",
    ),
    "yolo_object_detection_confidence": (
        "YOLO object/person features",
        "YOLOv8 object detector",
        "Average object detection confidence across sampled frames.",
    ),
    "yolo_human_interaction_intensity": (
        "YOLO object/person features",
        "YOLOv8 object detector",
        "Interaction proxy derived from people count and object count.",
    ),
    "emotion_face_count_mean": (
        "Face/emotion proxy features",
        "OpenCV Haar face detector",
        "Average number of faces detected in sampled frames.",
    ),
    "emotion_face_count_max": (
        "Face/emotion proxy features",
        "OpenCV Haar face detector",
        "Maximum number of faces detected in any sampled frame.",
    ),
    "emotion_angry_face_ratio_proxy": (
        "Face/emotion proxy features",
        "Face-region contrast proxy",
        "Lightweight proxy for intense facial appearance; not a real emotion classifier.",
    ),
    "emotion_fear_score_proxy": (
        "Face/emotion proxy features",
        "Face-region contrast proxy",
        "Lightweight fear/emotion intensity proxy; can be replaced by DeepFace or FER.",
    ),
    "emotion_stress_score_proxy": (
        "Face/emotion proxy features",
        "Face count and contrast instability",
        "Stress proxy from facial contrast and face-count instability.",
    ),
    "emotion_aggressive_expression_score_proxy": (
        "Face/emotion proxy features",
        "Face-region contrast proxy",
        "Aggressive-expression proxy; not a full DeepFace/FER classifier.",
    ),
    "emotion_instability_proxy": (
        "Face/emotion proxy features",
        "Frame-level face count variation",
        "Variation in detected face count across sampled frames.",
    ),
    "pose_instability_proxy": (
        "Pose/activity proxy features",
        "Frame-to-frame RGB difference",
        "Variation in motion across sampled frames.",
    ),
    "pose_velocity_proxy": (
        "Pose/activity proxy features",
        "Frame-to-frame RGB difference",
        "Average motion speed proxy across sampled frames.",
    ),
    "hand_leg_motion_proxy": (
        "Pose/activity proxy features",
        "Upper/lower frame motion difference",
        "Approximate body-region motion proxy from sampled frame differences.",
    ),
    "temporal_pose_consistency_proxy": (
        "Pose/activity proxy features",
        "Frame-to-frame RGB difference",
        "Stability score derived from motion variation.",
    ),
    "pose_movement_intensity": (
        "Pose/activity proxy features",
        "Frame-to-frame RGB difference",
        "Body movement intensity proxy from sampled-frame differences.",
    ),
    "pose_aggressive_movement_score": (
        "Pose/activity proxy features",
        "Frame-to-frame RGB difference",
        "Aggressive movement proxy from motion velocity and instability.",
    ),
    "pose_suspicious_activity_probability": (
        "Pose/activity proxy features",
        "Frame-to-frame RGB difference",
        "Suspicious activity probability proxy derived from movement intensity.",
    ),
    "fusion_video_activity_score": (
        "Fusion features",
        "Motion and activity descriptors",
        "Combined motion and activity score.",
    ),
    "fusion_temporal_change_score": (
        "Fusion features",
        "Motion and transition descriptors",
        "Motion score adjusted by scene transition frequency.",
    ),
    "fusion_visual_quality_score": (
        "Fusion features",
        "Blur/sharpness descriptor",
        "Visual quality score copied from blur/sharpness extraction.",
    ),
    "fusion_final_video_confidence_proxy": (
        "Fusion features",
        "Motion/activity/transition descriptors",
        "Simple confidence proxy derived from multiple video activity signals.",
    ),
    "fusion_unsafe_activity_score": (
        "Fusion features",
        "Object, emotion, pose, and transition descriptors",
        "Unsafe activity score combining object, emotion, pose, and temporal signals.",
    ),
    "fusion_violence_probability_proxy": (
        "Fusion features",
        "Object, emotion, and pose descriptors",
        "Violence probability proxy from weapon, emotion, and pose signals.",
    ),
    "fusion_final_multimodal_feature_score": (
        "Fusion features",
        "Final multimodal feature fusion",
        "Final combined multimodal score from motion, object, emotion, pose, and activity features.",
    ),
}

class ResNet18Embedder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.features = torch.nn.Sequential(*list(backbone.children())[:-1])
        self.output_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return torch.flatten(x, 1)


class MobileNetEmbedder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.output_dim = 576

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


class MobileNetV2Embedder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backbone = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        self.features = backbone.features
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.output_dim = 1280

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a multi-model feature dataset from safe and unsafe videos."
    )
    parser.add_argument("--safe-dir", default="Dataset/Safe", help="Root folder containing safe videos.")
    parser.add_argument("--unsafe-dir", default="Dataset/Unsafe", help="Root folder containing unsafe videos.")
    parser.add_argument("--output", default="Feature_Files/video_features_advanced.csv", help="Output CSV path.")
    parser.add_argument(
        "--cnn-model",
        choices=("resnet18", "mobilenet_v2", "mobilenet_v3_small"),
        default="resnet18",
        help="Pretrained CNN backbone for visual embeddings.",
    )
    parser.add_argument(
        "--seconds-per-frame",
        type=float,
        default=1.0,
        help="Extract one frame every N seconds.",
    )
    parser.add_argument(
        "--max-videos-per-class",
        type=int,
        default=None,
        help="Limit videos per class for quick experiments.",
    )
    parser.add_argument(
        "--pooling",
        choices=("mean", "max"),
        default="mean",
        help="How to aggregate frame embeddings into a single video embedding.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device used for CNN inference.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip videos that already exist in the output CSV.",
    )
    parser.add_argument(
        "--yolo-weights",
        default="Models/yolov8n.pt",
        help="YOLOv8 weights used for object/person/crowd features.",
    )
    parser.add_argument(
        "--skip-yolo",
        action="store_true",
        help="Skip YOLO object/person feature extraction.",
    )
    parser.add_argument(
        "--skip-proxy-features",
        action="store_true",
        help="Skip face/emotion and pose/activity proxy features.",
    )
    parser.add_argument(
        "--feature-dictionary",
        default="Feature_Files/feature_dictionary.csv",
        help="Output CSV that explains every feature column.",
    )
    return parser.parse_args()


def iter_video_files(folder: Path, limit: int | None) -> list[Path]:
    videos = sorted(
        path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    return videos[:limit] if limit is not None else videos


def select_device(device_name: str) -> str:
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_name


def load_visual_model(cnn_model: str, device: str) -> tuple[torch.nn.Module, object, int]:
    if cnn_model == "mobilenet_v2":
        weights = MobileNet_V2_Weights.DEFAULT
        model = MobileNetV2Embedder()
        preprocess = weights.transforms()
        return model.to(device).eval(), preprocess, model.output_dim

    if cnn_model == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.DEFAULT
        model = MobileNetEmbedder()
        preprocess = weights.transforms()
        return model.to(device).eval(), preprocess, model.output_dim

    weights = ResNet18_Weights.DEFAULT
    model = ResNet18Embedder()
    preprocess = weights.transforms()
    return model.to(device).eval(), preprocess, model.output_dim


def extract_visual_features(
    video_path: Path,
    model: torch.nn.Module,
    preprocess: object,
    output_dim: int,
    device: str,
    seconds_per_frame: float,
    pooling: str,
) -> tuple[np.ndarray, dict[str, float]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_seconds = total_frames / fps if fps else 0.0
    sample_interval = max(int(round(fps * seconds_per_frame)), 1)

    frame_index = 0
    embeddings: list[np.ndarray] = []

    with torch.inference_mode():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % sample_interval == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb)
                tensor = preprocess(pil_image).unsqueeze(0).to(device)
                embedding = model(tensor).squeeze(0).detach().cpu().numpy()
                embeddings.append(embedding)

            frame_index += 1

    cap.release()

    if not embeddings:
        aggregated = np.zeros(output_dim, dtype=np.float32)
    else:
        stacked = np.stack(embeddings)
        aggregated = stacked.mean(axis=0) if pooling == "mean" else stacked.max(axis=0)

    metadata = {
        "frames_sampled": len(embeddings),
        "video_fps": round(fps, 2),
        "video_duration_seconds": round(duration_seconds, 2),
    }
    return aggregated, metadata


def sample_proxy_frames(video_path: Path, seconds_per_sample: float, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return np.empty((0, size[1], size[0], 3), dtype=np.float32)

    fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    interval = max(int(round(fps * seconds_per_sample)), 1)
    frame_index = 0
    frames: list[np.ndarray] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, size)
            frames.append(resized.astype(np.float32) / 255.0)
        frame_index += 1

    cap.release()
    if not frames:
        return np.empty((0, size[1], size[0], 3), dtype=np.float32)
    return np.stack(frames)


def build_row(
    *,
    video_path: Path,
    label: int,
    class_name: str,
    visual_embedding: np.ndarray,
    visual_metadata: dict[str, float],
    extra_features: dict[str, float] | None = None,
) -> dict[str, str | int | float]:
    row: dict[str, str | int | float] = {
        "video_name": video_path.name,
        "video_path": str(video_path),
        "class_name": class_name,
        "label": label,
    }
    row.update(visual_metadata)
    if extra_features:
        row.update(extra_features)

    for index, value in enumerate(visual_embedding):
        row[cnn_feature_name(index)] = float(value)

    return row


def build_extra_features(
    *,
    video_path: Path,
    seconds_per_sample: float,
    yolo_model: object | None,
    include_proxy_features: bool,
) -> dict[str, float]:
    features = extract_classical_video_features(video_path, seconds_per_sample=seconds_per_sample)

    if yolo_model is not None:
        features.update(
            extract_yolo_features(
                video_path,
                yolo_model,
                seconds_per_sample=seconds_per_sample,
            )
        )

    if include_proxy_features:
        frames = sample_proxy_frames(video_path, seconds_per_sample=seconds_per_sample)
        features.update(extract_emotion_proxy_features(frames))
        features.update(extract_pose_proxy_features(frames))

    return add_video_fusion_features(features)


def write_feature_dictionary(output_dim: int, path: Path) -> None:
    rows: list[dict[str, str]] = []
    for feature_name, (feature_group, source, explanation) in EXPLICIT_FEATURE_EXPLANATIONS.items():
        rows.append(
            {
                "feature_name": feature_name,
                "feature_group": feature_group,
                "source": source,
                "explanation": explanation,
            }
        )

    cnn_groups = [
        ("edge_texture_feature_", "Edges, textures, contrast, and low-level appearance patterns"),
        ("object_body_feature_", "Objects, people, body parts, and visual structures"),
        ("scene_activity_feature_", "Scene layout, activity context, and environment patterns"),
        ("safety_context_feature_", "High-level combined safety-context patterns"),
    ]
    for index in range(output_dim):
        feature_name = cnn_feature_name(index)
        feature_group = next(group for prefix, group in cnn_groups if feature_name.startswith(prefix))
        rows.append(
            {
                "feature_name": feature_name,
                "feature_group": feature_group,
                "source": "Pretrained CNN embedding from sampled video frames",
                "explanation": (
                    "Learned CNN visual pattern activation. It is not a manually named feature, "
                    "so it is grouped by visual abstraction instead of shown as visual_001."
                ),
            }
        )

    pd.DataFrame(rows).to_csv(path, index=False)


def expected_explicit_feature_names(*, include_yolo: bool, include_proxy_features: bool) -> set[str]:
    names = set(EXPLICIT_FEATURE_EXPLANATIONS)
    if not include_yolo:
        names = {name for name in names if not name.startswith("yolo_")}
    if not include_proxy_features:
        names = {
            name
            for name in names
            if not name.startswith(("emotion_", "pose_", "hand_leg_", "temporal_pose_"))
        }
    return names


def main() -> None:
    args = parse_args()
    device = select_device(args.device)

    visual_model, preprocess, output_dim = load_visual_model(args.cnn_model, device)
    yolo_model = None
    if not args.skip_yolo:
        from ultralytics import YOLO

        yolo_model = YOLO(args.yolo_weights)

    existing_df = None
    processed_paths: set[str] = set()
    output_path = Path(args.output)

    if args.resume and output_path.exists():
        existing_df = pd.read_csv(output_path)
        expected_columns = expected_explicit_feature_names(
            include_yolo=not args.skip_yolo,
            include_proxy_features=not args.skip_proxy_features,
        )
        if expected_columns.issubset(existing_df.columns) and "video_path" in existing_df.columns:
            processed_paths = set(existing_df["video_path"].astype(str))
            print(f"Resuming from {output_path} with {len(processed_paths)} existing rows.")
        else:
            missing_columns = sorted(expected_columns - set(existing_df.columns))
            existing_df = None
            print(
                f"{output_path} is missing {len(missing_columns)} explicit feature columns, "
                "so videos will be reprocessed to rebuild the full feature dataset."
            )

    datasets = [
        (0, "safe", Path(args.safe_dir)),
        (1, "unsafe", Path(args.unsafe_dir)),
    ]

    rows: list[dict[str, str | int | float]] = []

    for label, class_name, folder in datasets:
        if not folder.exists():
            print(f"Skipping missing folder: {folder}")
            continue

        videos = iter_video_files(folder, args.max_videos_per_class)
        print(f"Found {len(videos)} {class_name} videos in {folder}")

        for index, video_path in enumerate(videos, start=1):
            if str(video_path) in processed_paths:
                print(f"[{index}/{len(videos)}] Skipped {video_path.name} (already processed)")
                continue

            try:
                visual_embedding, visual_metadata = extract_visual_features(
                    video_path=video_path,
                    model=visual_model,
                    preprocess=preprocess,
                    output_dim=output_dim,
                    device=device,
                    seconds_per_frame=args.seconds_per_frame,
                    pooling=args.pooling,
                )
                extra_features = build_extra_features(
                    video_path=video_path,
                    seconds_per_sample=args.seconds_per_frame,
                    yolo_model=yolo_model,
                    include_proxy_features=not args.skip_proxy_features,
                )
                rows.append(
                    build_row(
                        video_path=video_path,
                        label=label,
                        class_name=class_name,
                        visual_embedding=visual_embedding,
                        visual_metadata=visual_metadata,
                        extra_features=extra_features,
                    )
                )
                print(f"[{index}/{len(videos)}] Processed {video_path.name}")
            except Exception as exc:
                print(f"[{index}/{len(videos)}] Failed {video_path.name}: {exc}")

    if existing_df is not None:
        new_df = pd.DataFrame(rows)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = pd.DataFrame(rows)

    if combined.empty:
        raise SystemExit("No rows were created. Check dataset folders, model downloads, and video files.")

    combined.to_csv(output_path, index=False)
    write_feature_dictionary(output_dim, Path(args.feature_dictionary))
    print(f"Saved {len(combined)} rows to {output_path}")
    print(f"Saved feature dictionary to {args.feature_dictionary}")


if __name__ == "__main__":
    main()
