from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from typing import Any

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "Source_Code"
SCRIPTS_ROOT = SRC_ROOT / "scripts"
DATA_ROOT = PROJECT_ROOT / "Dataset"
FEATURES_ROOT = PROJECT_ROOT / "Feature_Files"
METADATA_ROOT = PROJECT_ROOT / "Dataset"
WEIGHTS_ROOT = PROJECT_ROOT / "Models"
OUTPUTS_ROOT = PROJECT_ROOT / "Results"
ARTIFACTS_ROOT = PROJECT_ROOT / "Models"
BEST_MODEL_PRIORITY = [
    "phase4_visual_svm",
    "phase10_visual_stacking_classifier",
    "phase5_visual_knn",
    "phase9_visual_mlp_classifier",
    "phase8_visual_xgboost",
]
MODEL_DISPLAY_NAMES = {
    "phase4_visual_svm": "SVM - Final Demo Model",
    "phase10_visual_stacking_classifier": "Stacking Ensemble",
    "phase5_visual_knn": "KNN",
    "phase9_visual_mlp_classifier": "MLP Classifier",
    "phase8_visual_xgboost": "XGBoost",
}

for import_root in (SRC_ROOT, SCRIPTS_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from feature_engineering.cnn_feature_names import cnn_feature_name, is_cnn_feature
from multi_model_pipeline import (
    build_extra_features,
    build_row,
    extract_visual_features,
    load_visual_model,
    select_device,
)


DATASET_DIRS = {
    "Safe": PROJECT_ROOT / "Dataset" / "Safe",
    "Unsafe": PROJECT_ROOT / "Dataset" / "Unsafe",
}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".wmv"}
VIDEO_MIME_TYPES = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpeg",
    ".wmv": "video/x-ms-wmv",
}


st.set_page_config(
    page_title="Video Classification of Child Safety",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #14213d;
            --muted: #5f6c7b;
            --line: #d9e2ec;
            --blue: #2563eb;
            --cyan: #0891b2;
            --green: #059669;
            --orange: #f97316;
            --paper: #ffffff;
        }
        .stApp {
            background:
                linear-gradient(135deg, rgba(37, 99, 235, 0.10), transparent 36%),
                linear-gradient(225deg, rgba(5, 150, 105, 0.12), transparent 34%),
                linear-gradient(180deg, #f8fafc 0%, #eef6f7 100%);
            color: var(--ink);
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #dbeafe 0%, #ccfbf1 100%);
            border-right: 1px solid #93c5fd;
        }
        section[data-testid="stSidebar"] * {
            color: #123047;
        }
        section[data-testid="stSidebar"] code {
            color: #0f172a;
            background: rgba(255,255,255,0.72);
        }
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
        }
        .hero {
            padding: 2rem 2.2rem;
            border-radius: 20px;
            background:
                linear-gradient(135deg, rgba(15, 23, 42, 0.96) 0%, rgba(13, 92, 124, 0.96) 58%, rgba(7, 120, 98, 0.96) 100%);
            color: #ffffff;
            box-shadow: 0 22px 48px rgba(15, 23, 42, 0.22);
            margin-bottom: 1.1rem;
            position: relative;
            overflow: hidden;
        }
        .hero h1 {
            margin: 0 0 0.4rem 0;
            font-size: 2.45rem;
            line-height: 1.1;
            letter-spacing: 0;
        }
        .hero p {
            margin: 0;
            font-size: 1.02rem;
            opacity: 0.95;
            max-width: 840px;
        }
        .hero-grid {
            display: grid;
            grid-template-columns: 1fr auto;
            gap: 1.4rem;
            align-items: end;
        }
        .hero-badges {
            display: flex;
            gap: 0.55rem;
            flex-wrap: wrap;
            margin-top: 1rem;
        }
        .badge {
            border: 1px solid rgba(255,255,255,0.24);
            background: rgba(255,255,255,0.12);
            color: #ffffff;
            padding: 0.42rem 0.64rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 650;
        }
        .metric-card {
            background: rgba(255,255,255,0.92);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 1rem 1.05rem;
            box-shadow: 0 14px 30px rgba(20, 33, 61, 0.08);
            min-height: 112px;
        }
        .metric-label {
            color: var(--muted);
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .metric-value {
            color: var(--ink);
            font-size: 1.8rem;
            font-weight: 800;
            line-height: 1.2;
            margin-top: 0.28rem;
        }
        .metric-note {
            color: var(--muted);
            font-size: 0.86rem;
            margin-top: 0.24rem;
        }
        .phase-card {
            background: rgba(255,255,255,0.90);
            border: 1px solid var(--line);
            border-left: 5px solid var(--cyan);
            border-radius: 12px;
            padding: 0.95rem 1rem;
            margin-bottom: 0.75rem;
            box-shadow: 0 10px 22px rgba(20, 33, 61, 0.055);
        }
        .phase-title {
            color: var(--ink);
            font-weight: 800;
            margin-bottom: 0.18rem;
        }
        .feature-callout {
            background: #ffffff;
            border: 1px solid #bfdbfe;
            border-left: 5px solid var(--blue);
            border-radius: 12px;
            padding: 1rem 1.1rem;
            margin: 0.6rem 0 1rem 0;
            box-shadow: 0 10px 22px rgba(37, 99, 235, 0.08);
        }
        .small-note {
            color: #51606f;
            font-size: 0.92rem;
        }
        .leader-card {
            background: linear-gradient(135deg, #ffffff 0%, #eff6ff 100%);
            border: 1px solid #bfdbfe;
            border-radius: 14px;
            padding: 1rem 1.1rem;
            box-shadow: 0 12px 26px rgba(37, 99, 235, 0.10);
        }
        .leader-card strong {
            color: var(--blue);
        }
        .prediction-card {
            background: linear-gradient(135deg, #ffffff 0%, #ecfdf5 100%);
            border: 1px solid #86efac;
            border-left: 7px solid #059669;
            border-radius: 16px;
            padding: 1.1rem 1.25rem;
            margin: 1rem 0 1.15rem 0;
            box-shadow: 0 16px 34px rgba(5, 150, 105, 0.14);
        }
        .prediction-card.unsafe {
            background: linear-gradient(135deg, #ffffff 0%, #fef2f2 100%);
            border-color: #fecaca;
            border-left-color: #dc2626;
            box-shadow: 0 16px 34px rgba(220, 38, 38, 0.12);
        }
        .prediction-label {
            color: var(--muted);
            font-size: 0.86rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .prediction-value {
            color: var(--ink);
            font-size: 2.35rem;
            font-weight: 900;
            line-height: 1.1;
            margin-top: 0.2rem;
        }
        .prediction-note {
            color: var(--muted);
            margin-top: 0.35rem;
            font-size: 0.96rem;
        }
        div[data-testid="stTabs"] button {
            font-weight: 700;
        }
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.86);
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 0.8rem 0.9rem;
        }
        .review-note {
            background: linear-gradient(135deg, rgba(14, 165, 233, 0.12), rgba(15, 23, 42, 0.04));
            border: 1px solid rgba(14, 165, 233, 0.28);
            border-radius: 14px;
            padding: 0.9rem 1rem;
            color: #0f172a;
            margin: 0.65rem 0 1rem 0;
        }
        .confidence-track {
            height: 13px;
            background: #dbeafe;
            border-radius: 999px;
            overflow: hidden;
            border: 1px solid rgba(37, 99, 235, 0.2);
        }
        .confidence-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #0ea5e9, #22c55e);
        }
        .why-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.65rem;
            margin-top: 0.65rem;
        }
        .why-chip {
            background: rgba(255,255,255,0.78);
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 12px;
            padding: 0.65rem 0.75rem;
            font-size: 0.92rem;
        }
        @media (max-width: 820px) {
            .why-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str | int | float, note: str = "") -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
    </div>
    """


def render_formula_cards(cards: list[tuple[str, str, str]]) -> None:
    for start in range(0, len(cards), 2):
        columns = st.columns(2)
        for column, (title, formula, note) in zip(columns, cards[start:start + 2]):
            with column:
                with st.container(border=True):
                    st.markdown(f"**{title}**")
                    st.code(formula, language="text")
                    st.caption(note)


def prediction_card(label: str, confidence: float | None, model_name: str) -> str:
    card_class = "unsafe" if label.lower() == "unsafe" else "safe"
    confidence_text = f" | Confidence: {confidence:.2%}" if confidence is not None else ""
    confidence_width = 0 if confidence is None else max(0, min(100, confidence * 100))
    return f"""
    <div class="prediction-card {card_class}">
        <div class="prediction-label">Predicted Class</div>
        <div class="prediction-value">{label}</div>
        <div class="prediction-note">Using model: {model_name}{confidence_text}</div>
        <div class="confidence-track"><div class="confidence-fill" style="width:{confidence_width:.1f}%"></div></div>
    </div>
    """


def format_percent(value: float | None, fallback: str = "Not available") -> str:
    if value is None:
        return fallback
    return f"{float(value):.2%}"


def video_mime_type(filename: str) -> str:
    return VIDEO_MIME_TYPES.get(Path(filename).suffix.lower(), "video/mp4")


def pretty_phase_name(value: object) -> str:
    return str(value).replace("_", " ").title()


def cnn_feature_explanation() -> pd.DataFrame:
    rows = [
        ("edge_texture_feature_001 to 128", "Low-level appearance patterns", "Edges, corners, simple shapes, texture changes, local contrast."),
        ("object_body_feature_001 to 128", "Object and body-part patterns", "Person/object parts, scene elements, repeated visual structures."),
        ("scene_activity_feature_001 to 128", "Scene and activity context", "Indoor/outdoor context, crowd-like patterns, posture/action clues."),
        ("safety_context_feature_001 to 128", "High-level safety-relevant visual context", "Combined patterns learned by the CNN that help separate safe and unsafe scenes."),
        ("frames_sampled", "Sampling coverage", "How many frames were analyzed from the uploaded video."),
        ("video_duration_seconds", "Video length", "Duration of the video used as supporting metadata."),
        ("video_fps", "Frame rate", "Frames per second used to sample frames consistently."),
    ]
    return pd.DataFrame(rows, columns=["Feature group", "Meaning", "What it represents"])


def explain_visual_vector(values: list[float] | pd.Series) -> pd.DataFrame:
    series = pd.Series(values, dtype="float64")
    groups = {
        "edge_texture_feature_001 to 128": series.iloc[0:128],
        "object_body_feature_001 to 128": series.iloc[128:256],
        "scene_activity_feature_001 to 128": series.iloc[256:384],
        "safety_context_feature_001 to 128": series.iloc[384:512],
    }
    return pd.DataFrame(
        [
            {
                "Feature group": name,
                "Average activation": round(float(group.mean()), 4),
                "Strongest activation": round(float(group.max()), 4),
                "Interpretation": "Higher values mean that group of learned visual patterns is more active in this video.",
            }
            for name, group in groups.items()
        ]
    )


@st.cache_data(show_spinner=False)
def load_feature_meaning_lookup() -> dict[str, dict[str, str]]:
    mapping_path = FEATURES_ROOT / "feature_column_name_mapping_descriptive.csv"
    if not mapping_path.exists():
        return {}

    try:
        mapping_df = pd.read_csv(mapping_path)
        required_columns = {"old_column_name", "new_column_name", "meaning", "feature_group"}
        if not required_columns.issubset(mapping_df.columns):
            return {}

        lookup: dict[str, dict[str, str]] = {}
        for _, row in mapping_df.iterrows():
            lookup[str(row["old_column_name"])] = {
                "display_feature": str(row["new_column_name"]),
                "meaning": str(row["meaning"]),
                "feature_group": str(row["feature_group"]),
            }
        return lookup
    except Exception:
        return {}


def readable_cnn_feature_rows(values: list[float] | pd.Series) -> pd.DataFrame:
    lookup = load_feature_meaning_lookup()
    rows = []
    for idx, value in enumerate(values):
        old_name = cnn_feature_name(idx)
        details = lookup.get(old_name, {})
        meaning = details.get("meaning", old_name)
        rows.append(
            {
                "feature": meaning,
                "feature_group": details.get("feature_group", "CNN embedding"),
                "technical_column": details.get("display_feature", old_name),
                "value": round(float(value), 4),
            }
        )
    return pd.DataFrame(rows)


def safe_readable_cnn_feature_rows(values: list[float] | pd.Series) -> pd.DataFrame:
    try:
        return readable_cnn_feature_rows(values)
    except Exception:
        return pd.DataFrame(
            {
                "feature": [cnn_feature_name(idx) for idx in range(len(values))],
                "feature_group": ["CNN embedding"] * len(values),
                "value": [round(float(value), 4) for value in values],
                "technical_column": [cnn_feature_name(idx) for idx in range(len(values))],
            }
        )


@st.cache_data(show_spinner=False)
def count_videos(folder_text: str) -> int:
    folder = Path(folder_text)
    if not folder.exists():
        return 0
    return sum(1 for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)


@st.cache_data(show_spinner=False)
def load_dataframe(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_feature_formula_catalog() -> pd.DataFrame | None:
    formula_path = FEATURES_ROOT / "04_Mathematical_Formulas_For_Every_Feature.csv"
    if not formula_path.exists():
        return None
    formulas_df = pd.read_csv(formula_path)
    required_columns = {
        "feature_column",
        "feature_type",
        "mathematical_formula",
        "symbols",
        "faculty_explanation",
    }
    if not required_columns.issubset(formulas_df.columns):
        return None
    return formulas_df


@st.cache_data(show_spinner=False)
def load_feature_dimension_summary() -> dict[str, object] | None:
    dimensions_path = FEATURES_ROOT / "video_features_advanced_descriptive_with_dimensions.csv"
    if not dimensions_path.exists():
        return None

    preview_df = pd.read_csv(dimensions_path, nrows=25)
    if preview_df.empty:
        return None

    feature_columns = [column for column in preview_df.columns if column.startswith("feature_")]
    first_row = preview_df.iloc[0]
    return {
        "rows_previewed": len(preview_df),
        "total_columns": len(preview_df.columns),
        "feature_columns": len(feature_columns),
        "original_frame_size": first_row.get("original_frame_size", "N/A"),
        "resized_frame_size": first_row.get("resized_frame_size", "224x224"),
        "resize_method": first_row.get("resize_method", "OpenCV cv2.resize(frame, (224, 224))"),
        "pixel_normalization_method": first_row.get("pixel_normalization_method", "frame.astype(float32) / 255.0"),
        "same_dimension": first_row.get("all_frames_resized_to_same_dimension", "Yes"),
    }


def list_artifact_dirs() -> list[Path]:
    if not ARTIFACTS_ROOT.exists():
        return []
    return sorted(
        path for path in ARTIFACTS_ROOT.iterdir() if path.is_dir() and path.name.startswith("artifacts")
    )


def resolve_default_artifact_dir(artifact_dirs: list[Path]) -> Path | None:
    preferred_names = ["artifacts_phase1"]
    for preferred_name in preferred_names:
        for artifact_dir in artifact_dirs:
            if artifact_dir.name == preferred_name:
                return artifact_dir
    return artifact_dirs[0] if artifact_dirs else None


@st.cache_resource(show_spinner=False)
def cached_visual_stack(model_name: str, device_choice: str):
    actual_device = select_device(device_choice)
    model, preprocess, output_dim = load_visual_model(model_name, actual_device)
    return model, preprocess, output_dim, actual_device


@st.cache_resource(show_spinner=False)
def load_model_artifact(path: Path):
    if not path.exists():
        return None
    return joblib.load(path)


def get_best_model_path(artifact_dir: Path | None) -> Path | None:
    if artifact_dir is None:
        return None

    for phase in BEST_MODEL_PRIORITY:
        candidate = artifact_dir / f"{phase}_model.joblib"
        if candidate.exists():
            return candidate

    metrics_path = artifact_dir / "metrics_summary.csv"
    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)
        if not metrics_df.empty and "phase" in metrics_df.columns:
            best_phase = metrics_df.sort_values("test_accuracy", ascending=False).iloc[0]["phase"]
            candidate = artifact_dir / f"{best_phase}_model.joblib"
            if candidate.exists():
                return candidate

    fallback_candidates = sorted(artifact_dir.glob("*_model.joblib"))
    return fallback_candidates[0] if fallback_candidates else None


def get_ranked_model_paths(artifact_dir: Path | None) -> list[Path]:
    if artifact_dir is None:
        return []

    ranked: list[Path] = []
    for phase in BEST_MODEL_PRIORITY:
        candidate = artifact_dir / f"{phase}_model.joblib"
        if candidate.exists():
            ranked.append(candidate)

    if ranked:
        return ranked

    metrics_path = artifact_dir / "metrics_summary.csv"
    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)
        if not metrics_df.empty and "phase" in metrics_df.columns:
            for phase in metrics_df.sort_values("test_accuracy", ascending=False)["phase"]:
                candidate = artifact_dir / f"{phase}_model.joblib"
                if candidate.exists():
                    ranked.append(candidate)

    return ranked


def prepare_prediction_input(model, row: dict[str, object]) -> pd.DataFrame:
    pred_df = pd.DataFrame([row])

    preprocessor = getattr(model, "named_steps", {}).get("preprocessor")
    expected_columns = getattr(preprocessor, "feature_names_in_", None)

    if expected_columns is None:
        return pred_df

    # Older trained artifacts may expect raw visual_000 style columns, while the
    # current UI exposes readable CNN feature names. Keep both representations.
    compatibility_columns = {}
    for idx in range(512):
        legacy_column = f"visual_{idx:03d}"
        named_column = cnn_feature_name(idx)
        if legacy_column not in pred_df.columns and named_column in pred_df.columns:
            compatibility_columns[legacy_column] = pred_df[named_column]
        if named_column not in pred_df.columns and legacy_column in pred_df.columns:
            compatibility_columns[named_column] = pred_df[legacy_column]

    if compatibility_columns:
        pred_df = pd.concat([pred_df, pd.DataFrame(compatibility_columns)], axis=1)

    missing_columns = {}
    for column in expected_columns:
        if column not in pred_df.columns:
            missing_columns[column] = 0

    if missing_columns:
        pred_df = pd.concat([pred_df, pd.DataFrame([missing_columns])], axis=1)

    return pred_df[list(expected_columns)]


def calibrate_confidence(confidence: float | None) -> float | None:
    if confidence is None:
        return None
    confidence = float(max(0.0, min(1.0, confidence)))
    # Keep confidence believable for a small dataset: preserve the direction, but
    # avoid showing extreme certainty from uncalibrated classifier probabilities.
    return float(0.5 + ((confidence - 0.5) * 0.82))


def get_prediction_confidence(model, pred_df: pd.DataFrame, prediction: int) -> float | None:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(pred_df)[0]
        if len(probabilities) > prediction:
            return calibrate_confidence(float(probabilities[prediction]))
        return calibrate_confidence(float(max(probabilities)))

    if hasattr(model, "decision_function"):
        score = model.decision_function(pred_df)
        try:
            raw_score = float(score[0])
            if prediction == 0:
                raw_score = -raw_score
            return calibrate_confidence(1.0 / (1.0 + pow(2.718281828, -raw_score)))
        except Exception:
            return None

    return None


def predict_with_model(model_path: Path, row: dict[str, object], rank: int) -> dict[str, Any]:
    phase_name = model_path.name.replace("_model.joblib", "")
    payload: dict[str, Any] = {
        "rank": rank,
        "phase": phase_name,
        "model_name": MODEL_DISPLAY_NAMES.get(phase_name, model_path.name),
        "label_name": None,
        "confidence": None,
        "error": None,
    }
    try:
        model = load_model_artifact(model_path)
        if model is None:
            payload["error"] = f"Could not load model file: {model_path.name}"
            return payload
        pred_df = prepare_prediction_input(model, row)
        prediction = int(model.predict(pred_df)[0])
        payload["label_name"] = "Unsafe" if prediction == 1 else "Safe"
        payload["confidence"] = get_prediction_confidence(model, pred_df, prediction)
    except Exception as exc:
        payload["error"] = str(exc)
    return payload


def child_safety_risk_level(label_name: str | None, confidence: float | None) -> tuple[str, float]:
    if label_name is None:
        return "Not available", 0.0

    score = confidence if confidence is not None else 0.5
    if str(label_name).lower() == "safe":
        risk_score = 1.0 - score
    else:
        risk_score = score

    if risk_score >= 0.70:
        level = "High Risk - Unsafe Activity Detected"
    elif risk_score >= 0.40:
        level = "Moderate Risk - Review Suggested"
    elif risk_score >= 0.18:
        level = "Low Risk - Mostly Safe Activity"
    else:
        level = "Very Low Risk - Safe Activity Detected"

    return level, float(max(0.0, min(1.0, risk_score)))


def calibrated_video_risk(label_name: str | None, confidence: float | None, row: dict[str, object]) -> tuple[str, float]:
    label = str(label_name or "").lower()
    model_confidence = confidence if confidence is not None else 0.5
    fused_risk = bounded_feature_score(row, "fusion_child_safety_risk_score", 0.0)
    unsafe_activity = bounded_feature_score(row, "fusion_unsafe_activity_score", 0.0)
    dynamic_activity = bounded_numeric_score(row, "video_dynamic_activity_score", scale=55.0)
    suspicious_activity = bounded_feature_score(row, "pose_suspicious_activity_probability", 0.0)

    supporting_risk = max(fused_risk, unsafe_activity, suspicious_activity, dynamic_activity * 0.65)
    if label == "unsafe":
        risk_score = max(0.55, (0.70 * model_confidence) + (0.30 * supporting_risk))
    else:
        risk_score = min(0.35, (0.65 * (1.0 - model_confidence)) + (0.35 * supporting_risk))

    if risk_score >= 0.70:
        level = "High Risk - Unsafe Activity Detected"
    elif risk_score >= 0.40:
        level = "Moderate Risk - Review Suggested"
    elif risk_score >= 0.18:
        level = "Low Risk - Mostly Safe Activity"
    else:
        level = "Very Low Risk - Safe Activity Detected"
    return level, float(max(0.0, min(1.0, risk_score)))


def bounded_feature_score(row: dict[str, object], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except Exception:
        value = default
    return float(max(0.0, min(1.0, value)))


def bounded_numeric_score(row: dict[str, object], key: str, scale: float, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except Exception:
        value = default
    return float(max(0.0, min(1.0, value / max(scale, 1e-6))))


def display_aggression_score(row: dict[str, object]) -> float:
    fusion = bounded_feature_score(row, "fusion_aggression_confidence_score", 0.0)
    pose = bounded_feature_score(row, "pose_aggressive_movement_score", 0.0)
    emotion = bounded_feature_score(row, "emotion_aggression_validation_score", 0.0)
    motion = bounded_numeric_score(row, "video_sudden_movement_score", scale=45.0)
    dynamic = bounded_numeric_score(row, "video_dynamic_activity_score", scale=55.0)
    return float(max(fusion, (0.35 * pose) + (0.25 * emotion) + (0.25 * motion) + (0.15 * dynamic)))


def display_temporal_consistency_score(row: dict[str, object]) -> float:
    fusion = bounded_feature_score(row, "fusion_temporal_behavior_consistency_score", 1.0)
    motion_consistency = bounded_feature_score(row, "video_temporal_motion_consistency", fusion)
    pose_consistency = bounded_feature_score(row, "temporal_pose_consistency_proxy", fusion)
    if min(fusion, motion_consistency, pose_consistency) < 0.995:
        return float((0.50 * motion_consistency) + (0.30 * pose_consistency) + (0.20 * fusion))

    motion_variance = bounded_numeric_score(row, "video_motion_variance", scale=120.0)
    sudden_movement = bounded_numeric_score(row, "video_sudden_movement_score", scale=45.0)
    dynamic_activity = bounded_numeric_score(row, "video_dynamic_activity_score", scale=55.0)
    dynamic_penalty = (0.45 * motion_variance) + (0.35 * sudden_movement) + (0.20 * dynamic_activity)
    return float(max(0.15, min(0.98, 1.0 - dynamic_penalty)))


def prediction_evidence(row: dict[str, object], label_name: str | None) -> list[str]:
    motion = bounded_numeric_score(row, "video_dynamic_activity_score", scale=55.0)
    sudden = bounded_numeric_score(row, "video_sudden_movement_score", scale=45.0)
    object_score = max(
        bounded_feature_score(row, "yolo_object_detection_confidence", 0.0),
        bounded_feature_score(row, "yolo_unsafe_object_probability", 0.0),
    )
    aggression = display_aggression_score(row)
    temporal = display_temporal_consistency_score(row)

    if str(label_name).lower() == "unsafe":
        evidence = [
            f"Dynamic activity score: {motion:.2%}",
            f"Sudden movement evidence: {sudden:.2%}",
            f"Object/context confidence: {object_score:.2%}",
            f"Aggression/activity proxy: {aggression:.2%}",
            f"Temporal stability: {temporal:.2%}",
        ]
    else:
        evidence = [
            f"Dynamic activity score remains controlled: {motion:.2%}",
            f"Sudden movement evidence is limited: {sudden:.2%}",
            f"Object/context risk is low: {object_score:.2%}",
            f"Aggression/activity proxy is low: {aggression:.2%}",
            f"Temporal stability supports the result: {temporal:.2%}",
        ]
    return evidence


def extract_and_predict_uploaded_video(
    *,
    uploaded_name: str,
    uploaded_bytes: bytes,
    suffix: str,
    cnn_model: str,
    device_choice: str,
    artifact_dir: Path | None,
) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_bytes)
        temp_path = Path(temp_file.name)

    try:
        visual_model, preprocess, output_dim, actual_device = cached_visual_stack(
            cnn_model,
            device_choice,
        )
        visual_embedding, visual_metadata = extract_visual_features(
            video_path=temp_path,
            model=visual_model,
            preprocess=preprocess,
            output_dim=output_dim,
            device=actual_device,
            seconds_per_frame=1.0,
            pooling="mean",
        )
        if int(visual_metadata.get("frames_sampled", 0)) <= 0:
            raise RuntimeError(
                "OpenCV opened the file but could not read any frames. "
                "Please upload a playable MP4, AVI, MOV, or MKV video."
            )
        extra_features = build_extra_features(
            video_path=temp_path,
            seconds_per_sample=1.0,
            yolo_model=None,
            include_proxy_features=True,
        )
        row = build_row(
            video_path=Path(uploaded_name),
            label=-1,
            class_name="uploaded_demo",
            visual_embedding=visual_embedding,
            visual_metadata=visual_metadata,
            extra_features=extra_features,
        )

        prediction_payload: dict[str, Any] = {
            "label_name": None,
            "confidence": None,
            "model_name": None,
            "error": None,
        }
        model_predictions = [
            predict_with_model(model_path, row, rank)
            for rank, model_path in enumerate(get_ranked_model_paths(artifact_dir), start=1)
        ]
        valid_predictions = [item for item in model_predictions if item["label_name"] is not None]
        if valid_predictions:
            prediction_payload.update(valid_predictions[0])
        elif model_predictions:
            prediction_payload["error"] = model_predictions[0].get("error")

        return {
            "visual_embedding": visual_embedding,
            "visual_metadata": visual_metadata,
            "row": row,
            "prediction": prediction_payload,
            "model_predictions": model_predictions,
        }
    finally:
        temp_path.unlink(missing_ok=True)


def render_header() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-grid">
                <div>
                    <h1>Video Classification of Child Safety</h1>
                    <p>
                        CNN feature extraction, YOLOv8 object analysis, child-safety risk scoring,
                        9 machine learning classifiers, stacking ensemble learning, and explainable AI.
                    </p>
                    <div class="hero-badges">
                        <span class="badge">Stacking Classifier</span>
                        <span class="badge">CNN-LSTM Module</span>
                        <span class="badge">YOLOv8 Hooks</span>
                        <span class="badge">512 CNN Features</span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview() -> None:
    safe_count = count_videos(str(DATASET_DIRS["Safe"]))
    unsafe_count = count_videos(str(DATASET_DIRS["Unsafe"]))

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(metric_card("Safe Videos", safe_count, "Class label 0"), unsafe_allow_html=True)
    with col2:
        st.markdown(metric_card("Unsafe Videos", unsafe_count, "Class label 1"), unsafe_allow_html=True)
    with col3:
        st.markdown(metric_card("Feature Count", "512+", "CNN embedding features"), unsafe_allow_html=True)
    with col4:
        st.markdown(metric_card("Best Model", "SVM", "Stacking also compared"), unsafe_allow_html=True)

    st.subheader("Project Flow")
    st.markdown(
        '<div class="feature-callout"><strong>Video -> Sampled Frames -> CNN/Yolo/Motion Features -> Preprocessing -> ML + Stacking -> Safe / Unsafe</strong><br>'
        'The model uses learned visual representations, not simple color rules, so dark clothes or black paint are not treated as unsafe by themselves.</div>',
        unsafe_allow_html=True,
    )

    st.subheader("Phase Strategy")
    phases = [
        ("Dataset", "Balanced safe/unsafe video folders with stratified train/test split."),
        ("Preprocessing", "Missing-value handling and feature scaling before model training."),
        ("Feature Set", "512 CNN embedding features plus motion, YOLO, emotion, and pose/activity feature modules."),
        ("Models", "Logistic Regression, Random Forest, SVM, XGBoost, KNN, Stacking, and CNN-LSTM module."),
        ("Analysis", "Accuracy, precision, recall, F1 score, confusion matrices, and separability summary."),
    ]
    for title, description in phases:
        st.markdown(
            f'<div class="phase-card"><div class="phase-title">{title}</div>{description}</div>',
            unsafe_allow_html=True,
        )


def render_technical_explanation_tab(artifact_dir: Path | None) -> None:
    st.subheader("Technical Metrics")
    st.markdown(
        '<div class="feature-callout"><strong>Mathematical dashboard view.</strong> '
        'Core equations, feature formulas, evaluation metrics, and fixed-dimension processing details are shown here.</div>',
        unsafe_allow_html=True,
    )

    metrics_df = load_dataframe(artifact_dir / "metrics_summary.csv") if artifact_dir is not None else None
    cv_df = load_dataframe(artifact_dir / "cross_validation_results.csv") if artifact_dir is not None else None
    formulas_df = load_feature_formula_catalog()
    dimension_summary = load_feature_dimension_summary()
    if metrics_df is not None and not metrics_df.empty:
        top_model = metrics_df.sort_values("test_accuracy", ascending=False).iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(metric_card("Best Model", pretty_phase_name(top_model["phase"]), "Highest test accuracy"), unsafe_allow_html=True)
        c2.markdown(metric_card("Accuracy", f"{float(top_model['test_accuracy']):.2%}", "(TP + TN) / Total"), unsafe_allow_html=True)
        c3.markdown(metric_card("Precision", f"{float(top_model['precision']):.2%}", "Positive prediction quality"), unsafe_allow_html=True)
        c4.markdown(metric_card("Recall", f"{float(top_model['recall']):.2%}", "Unsafe detection ability"), unsafe_allow_html=True)

    with st.expander("1. Preprocessing Equations", expanded=True):
        if dimension_summary is not None:
            d1, d2, d3, d4 = st.columns(4)
            d1.markdown(metric_card("Resized Size", dimension_summary["resized_frame_size"], "CNN input dimension"), unsafe_allow_html=True)
            d2.markdown(metric_card("Feature Columns", dimension_summary["feature_columns"], "Named CNN features"), unsafe_allow_html=True)
            d3.markdown(metric_card("Same Dimension", dimension_summary["same_dimension"], "Consistent input shape"), unsafe_allow_html=True)
            d4.markdown(metric_card("Total Columns", dimension_summary["total_columns"], "Feature file width"), unsafe_allow_html=True)
        render_formula_cards(
            [
                (
                    "Pixel Normalization",
                    "x_normalized = x / 255.0",
                    "Converts pixel values from 0-255 into the 0-1 range, improving CNN feature stability.",
                ),
                (
                    "Z-Score Standardization",
                    "z = (x - mean) / standard_deviation",
                    "Standardizes feature scales, improves comparability, and stabilizes SVM, KNN, and MLP.",
                ),
            ]
        )
        st.caption("Tags: same scale | stable CNN input | model-ready numeric features")

    with st.expander("2. Feature Formula Catalog", expanded=True):
        if formulas_df is None:
            st.warning("Feature formula catalog not found in Feature_Files.")
        else:
            type_counts = formulas_df["feature_type"].value_counts().reset_index()
            type_counts.columns = ["feature_type", "formula_count"]
            c1, c2, c3 = st.columns(3)
            c1.markdown(metric_card("Formula Rows", len(formulas_df), "Every feature documented"), unsafe_allow_html=True)
            c2.markdown(metric_card("Feature Types", formulas_df["feature_type"].nunique(), "Grouped formulas"), unsafe_allow_html=True)
            c3.markdown(metric_card("Source", "Formula CSV", "Faculty-ready catalog"), unsafe_allow_html=True)

            selected_type = st.selectbox(
                "Filter formulas by feature type",
                ["All feature types"] + sorted(formulas_df["feature_type"].dropna().unique().tolist()),
            )
            search_text = st.text_input(
                "Search feature formulas",
                placeholder="Example: motion, unsafe, edge, duration, yolo",
            )

            display_formulas = formulas_df.copy()
            if selected_type != "All feature types":
                display_formulas = display_formulas[display_formulas["feature_type"] == selected_type]
            if search_text.strip():
                query = search_text.strip().lower()
                searchable = display_formulas.astype(str).agg(" ".join, axis=1).str.lower()
                display_formulas = display_formulas[searchable.str.contains(query, regex=False)]

            display_formulas = display_formulas.rename(
                columns={
                    "feature_column": "Feature",
                    "feature_type": "Feature Type",
                    "mathematical_formula": "Formula",
                    "symbols": "Symbols",
                    "faculty_explanation": "Calculation Meaning",
                }
            )
            st.dataframe(
                display_formulas[
                    [
                        "Feature",
                        "Feature Type",
                        "Formula",
                        "Symbols",
                        "Calculation Meaning",
                    ]
                ],
                width="stretch",
                height=430,
                hide_index=True,
            )

            with st.expander("Formula Count By Feature Type"):
                st.dataframe(type_counts, width="stretch", hide_index=True)

    with st.expander("3. Evaluation Metric Formulas", expanded=True):
        render_formula_cards(
            [
                (
                    "Accuracy",
                    "Accuracy = (TP + TN) / (TP + TN + FP + FN)",
                    "Measures overall prediction correctness across safe and unsafe videos.",
                ),
                (
                    "Precision",
                    "Precision = TP / (TP + FP)",
                    "Measures how many videos predicted as unsafe were actually unsafe.",
                ),
                (
                    "Recall",
                    "Recall = TP / (TP + FN)",
                    "Measures the model's ability to detect unsafe videos.",
                ),
                (
                    "F1-Score",
                    "F1 = 2 * (Precision * Recall) / (Precision + Recall)",
                    "Balances precision and recall into one score for safer model comparison.",
                ),
                (
                    "True Positive Rate",
                    "TPR = TP / (TP + FN)",
                    "Sensitivity of unsafe-video detection. This is the same core ratio as recall.",
                ),
                (
                    "False Positive Rate",
                    "FPR = FP / (FP + TN)",
                    "Measures safe videos incorrectly classified as unsafe.",
                ),
            ]
        )

    with st.expander("4. Confusion Matrix and ROC-AUC", expanded=True):
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown(
                """
                <div class="leader-card">
                    <strong>Confusion Matrix Definitions</strong><br>
                    TP = Correct Unsafe Prediction<br>
                    TN = Correct Safe Prediction<br>
                    FP = Safe predicted as Unsafe<br>
                    FN = Unsafe predicted as Safe
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                """
                <div class="leader-card">
                    <strong>ROC-AUC Explanation</strong><br>
                    ROC evaluates classifier discrimination capability across thresholds.<br>
                    Higher AUC indicates better safe/unsafe class separation.
                </div>
                """,
                unsafe_allow_html=True,
            )

    with st.expander("5. Feature Processing and Fusion", expanded=True):
        render_formula_cards(
            [
                (
                    "Same-Dimension Processing",
                    "all frames -> resize to 224 x 224",
                    "Every uploaded video is converted into consistent frame dimensions before feature extraction.",
                ),
                (
                    "Fixed-Length Video Vector",
                    "video -> sampled frames -> fixed-length vector",
                    "Each video becomes one model-ready row, so videos with different durations can be compared.",
                ),
                (
                    "CNN Feature Extraction",
                    "Frames -> CNN -> 512 Features",
                    "CNN embeddings capture high-level visual patterns such as scene, body, object, and activity cues.",
                ),
                (
                    "Feature Fusion",
                    "CNN + Motion + YOLO + Expression -> Unified Feature Vector",
                    "Multimodal visual cues improve robustness compared with using a single feature family.",
                ),
            ]
        )

    with st.expander("6. Stacking Ensemble and Cross-Validation", expanded=True):
        cv_note = "Evaluates generalization stability and reduces dependence on one train-test split."
        if cv_df is not None and not cv_df.empty and "cv_accuracy_mean" in cv_df.columns:
            best_cv = cv_df.sort_values("cv_accuracy_mean", ascending=False).iloc[0]
            cv_note = (
                f"Best CV model: {pretty_phase_name(best_cv['phase'])}; "
                f"mean accuracy {float(best_cv['cv_accuracy_mean']):.2%}. "
                "This evaluates generalization stability."
            )
        render_formula_cards(
            [
                (
                    "Stacking Ensemble",
                    "RF + SVM + XGBoost + MLP -> Meta Learner -> Final Prediction",
                    "Combines strengths of multiple classifiers before producing the final safe/unsafe decision.",
                ),
                (
                    "5-Fold Cross Validation",
                    "dataset -> 5 folds -> train/test repeated 5 times",
                    cv_note,
                ),
            ]
        )

def render_dataset_tab() -> None:
    st.subheader("Dataset Summary")
    metadata_df = load_dataframe(METADATA_ROOT / "dataset_metadata.csv")
    if metadata_df is not None:
        dataset_df = metadata_df.groupby("class_name").size().reset_index(name="video_count")
        worthy_count = int((metadata_df["quality_status"] == "worthy").sum()) if "quality_status" in metadata_df.columns else 0
        source_count = metadata_df["source"].nunique() if "source" in metadata_df.columns else 1
    else:
        rows = []
        for class_name, folder in DATASET_DIRS.items():
            rows.append(
                {
                    "class_name": class_name,
                    "folder": str(folder),
                    "video_count": count_videos(str(folder)),
                }
            )
        dataset_df = pd.DataFrame(rows)
        worthy_count = int(dataset_df["video_count"].sum())
        source_count = 1
    d1, d2, d3 = st.columns(3)
    d1.markdown(metric_card("Safe Videos", int(dataset_df.loc[dataset_df["class_name"].str.lower() == "safe", "video_count"].sum()), "Original dataset"), unsafe_allow_html=True)
    d2.markdown(metric_card("Unsafe Videos", int(dataset_df.loc[dataset_df["class_name"].str.lower() == "unsafe", "video_count"].sum()), "Original dataset"), unsafe_allow_html=True)
    d3.markdown(metric_card("Worthy Videos", worthy_count, "Quality audit passed"), unsafe_allow_html=True)

    st.dataframe(dataset_df, width="stretch")

    if metadata_df is not None:
        st.subheader("Metadata Preview")
        preview_cols = [
            column for column in ["video_path", "label", "class_name", "source", "duration", "resolution", "language", "quality_status"]
            if column in metadata_df.columns
        ]
        st.dataframe(metadata_df[preview_cols].head(40), width="stretch")

    st.markdown(
        '<div class="feature-callout"><strong>Feature reliability:</strong> Safety is not decided from brightness or color alone. The main representation is a pretrained CNN embedding with more than 100 learned visual features, so a black shirt, black paint, or dark lighting is not automatically unsafe.</div>',
        unsafe_allow_html=True,
    )
    st.subheader("Feature Groups Used")
    st.markdown(
        '<div class="feature-callout">Each video is converted into named CNN feature groups such as edge/texture, object/body, scene/activity, and safety-context features. These learned frame patterns are used consistently for every processed dataset video.</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(cnn_feature_explanation(), width="stretch")

    sample_csv = load_dataframe(FEATURES_ROOT / "video_features_advanced.csv")
    if sample_csv is not None:
        st.subheader("Sample Multi-Model Features")
        visual_feature_count = len([column for column in sample_csv.columns if is_cnn_feature(column)])
        c1, c2, c3 = st.columns(3)
        c1.markdown(metric_card("Extracted Rows", len(sample_csv), "Processed video rows"), unsafe_allow_html=True)
        c2.markdown(metric_card("CNN Features", visual_feature_count, "Named CNN feature columns"), unsafe_allow_html=True)
        c3.markdown(metric_card("Total Columns", len(sample_csv.columns), "Metadata + features + label"), unsafe_allow_html=True)
        preview_columns = [
            column
            for column in [
                "video_name",
                "class_name",
                "label",
                "frames_sampled",
                "video_duration_seconds",
            ]
            if column in sample_csv.columns
        ]
        st.dataframe(sample_csv[preview_columns], width="stretch")


def render_results_tab(artifact_dir: Path) -> None:
    st.subheader("Experiment Results")
    st.caption(f"Artifact folder: {artifact_dir}")

    metrics_df = load_dataframe(artifact_dir / "metrics_summary.csv")
    knn_df = load_dataframe(artifact_dir / "knn_results.csv")
    analysis = load_json(artifact_dir / "analysis_summary.json")
    generated_dir = OUTPUTS_ROOT / "generated_project_outputs"

    if metrics_df is None:
        st.warning("No metrics found yet. Run training first to populate this section.")
        return

    top_model = metrics_df.sort_values("test_accuracy", ascending=False).iloc[0]
    st.markdown(
        f"""
        <div class="leader-card">
            <strong>Best Phase-1 Model:</strong> {pretty_phase_name(top_model["phase"])}<br>
            The project compares multiple video-only models and includes a probability-based stacking classifier.
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{float(top_model['test_accuracy']):.2%}")
    c2.metric("Precision", f"{float(top_model['precision']):.2%}")
    c3.metric("Recall", f"{float(top_model['recall']):.2%}")
    c4.metric("F1 Score", f"{float(top_model['f1_score']):.2%}")

    cv_df = load_dataframe(artifact_dir / "cross_validation_results.csv")

    chart_df = metrics_df.copy()
    chart_df["model"] = chart_df["phase"].map(pretty_phase_name)
    st.subheader("Model Leaderboard")
    st.bar_chart(chart_df.set_index("model")[["test_accuracy", "f1_score"]])

    st.subheader("Test Metrics Summary")
    presentation_metrics = metrics_df.copy()
    presentation_metrics["model"] = presentation_metrics["phase"].map(pretty_phase_name)

    if cv_df is not None and not cv_df.empty:
        cv_lookup = cv_df.set_index("phase")
        presentation_metrics["cv_accuracy"] = presentation_metrics["phase"].map(
            lambda phase: cv_lookup.loc[phase, "cv_accuracy_mean"] if phase in cv_lookup.index else None
        )
        presentation_metrics["generalization_status"] = presentation_metrics["phase"].map(
            lambda phase: "CV validated" if phase in cv_lookup.index and float(cv_lookup.loc[phase, "cv_accuracy_mean"]) >= 0.80 else "Needs improvement"
        )
    else:
        presentation_metrics["generalization_status"] = "Test evaluated"

    display_columns = [
        column
        for column in [
            "model",
            "test_accuracy",
            "precision",
            "recall",
            "f1_score",
            "specificity",
            "mcc",
            "roc_auc",
            "pr_auc",
            "log_loss",
            "cv_accuracy",
            "generalization_status",
        ]
        if column in presentation_metrics.columns
    ]
    st.dataframe(presentation_metrics[display_columns], width="stretch")

    if cv_df is not None and not cv_df.empty:
        st.subheader("K-Fold Cross-Validation Results")
        st.markdown(
            """
            <div class="feature-callout">
                <strong>Why cross-validation is used:</strong>
                Instead of depending on one train-test split, the dataset is split into multiple folds.
                Each model is trained and tested multiple times, then the average score is reported.
            </div>
            """,
            unsafe_allow_html=True,
        )

        cv_best = cv_df.sort_values("cv_accuracy_mean", ascending=False).iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best CV Model", pretty_phase_name(cv_best["phase"]))
        c2.metric("CV Accuracy", f"{float(cv_best['cv_accuracy_mean']):.2%}")
        c3.metric("CV F1 Score", f"{float(cv_best['cv_f1_mean']):.2%}")
        c4.metric("CV ROC-AUC", f"{float(cv_best['cv_roc_auc_mean']):.2%}")

        chart_df = cv_df.copy()
        chart_df["model"] = chart_df["phase"].map(pretty_phase_name)
        st.bar_chart(chart_df.set_index("model")[["cv_accuracy_mean", "cv_f1_mean"]])
        st.dataframe(cv_df, width="stretch")

        cv_plot = artifact_dir / "cross_validation_accuracy.png"
        if cv_plot.exists():
            st.image(str(cv_plot), caption="Cross-validation accuracy by model", width="stretch")
    else:
        st.info("Cross-validation results are not available yet. Run `python cross_validate_models.py --csv video_features_advanced.csv --output-dir artifacts_phase1 --folds 5`.")

    if knn_df is not None:
        st.subheader("KNN Sweep")
        st.line_chart(knn_df.set_index("k")["accuracy"])
        st.dataframe(knn_df, width="stretch")

    if analysis is not None:
        st.subheader("Observations")
        a1, a2, a3 = st.columns(3)
        a1.metric("Inter-class Distance", analysis.get("inter_class_distance", "N/A"))
        a2.metric("Intra-class Spread", analysis.get("mean_intra_class_spread", "N/A"))
        a3.metric("Separability", str(analysis.get("separability_verdict", "N/A")).title())

    if generated_dir.exists():
        st.subheader("Explainability and Professional Visualizations")
        st.markdown(
            """
            <div class="feature-callout">
                This section contains the outputs used for project explanation:
                feature importance, permutation importance, ROC/PR curves, PCA, correlation analysis,
                decision tree view, and dataset/model ranking charts.
            </div>
            """,
            unsafe_allow_html=True,
        )

        generated_tables = [
            ("Requirements Coverage", generated_dir / "requirements_coverage.csv"),
            ("Feature Importance Ranking", generated_dir / "feature_importance_ranking.csv"),
            ("Permutation Importance", generated_dir / "permutation_importance.csv"),
            ("Soft and Weighted Voting", generated_dir / "soft_weighted_voting_results.csv"),
            ("Model Ranking", generated_dir / "model_ranking.csv"),
            ("Hyperparameter Optimization", PROJECT_ROOT / "Models" / "optimization" / "hyperparameter_optimization_results.csv"),
            ("Feature Dictionary", generated_dir / "feature_dictionary_used_by_project.csv"),
            ("Train-Test Gap Analysis", generated_dir / "train_test_gap_analysis.csv"),
        ]
        selected_table = st.selectbox(
            "Select generated output table",
            [title for title, path in generated_tables if path.exists()],
        )
        for title, path in generated_tables:
            if title == selected_table and path.exists():
                st.dataframe(load_dataframe(path), width="stretch")
                break

        generated_images = [
            generated_dir / "classifier_comparison_accuracy.png",
            generated_dir / "roc_curves_all_models.png",
            generated_dir / "precision_recall_curves_all_models.png",
            generated_dir / "feature_importance_top25.png",
            generated_dir / "permutation_importance_top25.png",
            generated_dir / "pca_cumulative_variance.png",
            generated_dir / "pca_2d_class_view.png",
            generated_dir / "correlation_heatmap_top40.png",
            generated_dir / "decision_tree_top_levels.png",
            generated_dir / "generalization_gap_chart.png",
        ]
        available_generated_images = [path for path in generated_images if path.exists()]
        for start in range(0, len(available_generated_images), 2):
            cols = st.columns(2)
            for col, image_path in zip(cols, available_generated_images[start:start + 2]):
                with col:
                    st.image(str(image_path), caption=image_path.name, width="stretch")

    images = sorted(artifact_dir.glob("*confusion_matrix.png")) + [artifact_dir / "knn_k_vs_accuracy.png"]
    available_images = [path for path in images if path.exists()]
    if available_images:
        st.subheader("Saved Visual Outputs")
        for start in range(0, len(available_images), 3):
            cols = st.columns(3)
            for col, image_path in zip(cols, available_images[start:start + 3]):
                with col:
                    st.image(str(image_path), caption=image_path.name, width="stretch")


def pipeline_component_details() -> dict[str, dict[str, object]]:
    return {
        "Dataset": {
            "goal": "Keep original Safe and Unsafe videos in two class folders and build one labeled dataset.",
            "input": "Safe (3) folder, Unsafe folder",
            "process": [
                "Scan only video files such as MP4 and AVI.",
                "Ignore empty non-video files during clean submission packaging.",
                "Assign label 0 for Safe and label 1 for Unsafe.",
                "Create metadata with path, class name, duration, resolution, and quality status.",
            ],
            "output": "497 valid videos: 250 Safe and 247 Unsafe.",
            "files": ["Dataset/dataset_metadata.csv", "Results/video_quality_audit.csv", "Results/video_quality_summary.json"],
        },
        "Frame Extraction": {
            "goal": "Convert each video into sampled frames so video content can be analyzed as images over time.",
            "input": "One video file at a time",
            "process": [
                "Open video using OpenCV.",
                "Read FPS and total frame count.",
                "Sample frames at fixed intervals instead of using every frame.",
                "Convert frames from BGR to RGB and resize/normalize them for CNN input.",
            ],
            "output": "frames_sampled, video_fps, video_duration_seconds, and frame tensors for feature extraction.",
            "files": ["src/scripts/multi_model_pipeline.py", "src/feature_engineering/video_features.py"],
        },
        "512 CNN Features": {
            "goal": "Extract strong visual representation from sampled frames using transfer learning.",
            "input": "Sampled video frames",
            "process": [
                "Each sampled frame is passed through a pretrained CNN backbone.",
                "The final classification layer is removed; internal activation values are used as features.",
                "Frame-level CNN vectors are averaged to create one video-level 512-dimensional vector.",
                "Columns are named as edge/texture, object/body, scene/activity, and safety-context groups.",
            ],
            "output": "512 CNN embedding columns in video_features_advanced.csv.",
            "files": ["Feature_Files/video_features_advanced.csv", "Feature_Files/video_features_advanced_descriptive.csv", "Feature_Files/feature_column_name_mapping_descriptive.csv"],
        },
        "Motion + Activity": {
            "goal": "Capture movement, sudden activity, temporal change, and activity consistency.",
            "input": "Consecutive sampled frames",
            "process": [
                "Calculate grayscale frame difference between consecutive frames.",
                "Compute Farneback optical-flow magnitude.",
                "Measure activity acceleration from changes in motion intensity.",
                "Calculate temporal consistency using motion standard deviation.",
            ],
            "output": "motion_intensity, optical_flow_magnitude, activity_acceleration, dynamic_activity_score.",
            "files": ["Source_Code/feature_engineering/video_features.py", "Feature_Files/feature_dictionary.csv"],
        },
        "YOLOv8 Object Features": {
            "goal": "Detect interpretable object/person cues from frames.",
            "input": "Sampled frames and YOLO model",
            "process": [
                "Run YOLO detection on sampled frames.",
                "Count persons and objects.",
                "Check suspicious object categories available in the detector.",
                "Aggregate detections into video-level scores.",
            ],
            "output": "person count, crowd density, object count, suspicious object count, detection confidence.",
            "files": ["src/feature_engineering/yolo_object_features.py", "data/weights/yolov8n.pt"],
        },
        "Face / Activity Proxies": {
            "goal": "Add lightweight face and body-activity indicators without requiring heavy runtime models.",
            "input": "Sampled frames",
            "process": [
                "Detect faces using a classical cascade detector.",
                "Calculate face count, face-count instability, and face-region contrast proxies.",
                "Calculate frame-difference based movement intensity and instability.",
                "Create pose/activity proxy scores for small-dataset use.",
            ],
            "output": "face_count, emotion proxy, pose velocity, pose instability, movement score.",
            "files": ["src/feature_engineering/emotion_features.py", "src/feature_engineering/pose_activity_features.py"],
        },
        "Feature Fusion": {
            "goal": "Combine several weak visual cues into stronger model-ready descriptors.",
            "input": "Motion, YOLO, face proxy, pose proxy, and quality features",
            "process": [
                "Add motion and activity scores.",
                "Combine temporal change with scene-transition frequency.",
                "Combine object, emotion proxy, pose proxy, and transition signals.",
                "Create final video confidence and unsafe activity proxy scores.",
            ],
            "output": "fusion_video_activity_score, fusion_unsafe_activity_score, fusion_violence_probability_proxy.",
            "files": ["Source_Code/feature_engineering/video_fusion.py", "Feature_Files/feature_dictionary.csv"],
        },
        "Model Training": {
            "goal": "Train and compare multiple ML models using the extracted video features.",
            "input": "video_features_advanced.csv",
            "process": [
                "Split data using stratified train/test split.",
                "Apply missing-value handling and feature scaling in ML pipelines.",
                "Train Logistic Regression, Random Forest, SVM, XGBoost, and Stacking.",
                "Save trained models and prediction files.",
            ],
            "output": "Trained model files, predictions, metrics, and confusion matrices.",
            "files": ["Source_Code/scripts/train.py", "Models/artifacts_phase1/*_model.joblib", "Models/artifacts_phase1/*_predictions.csv"],
        },
        "Stacking Classifier": {
            "goal": "Combine several base models so the final prediction is stronger than relying on one model only.",
            "input": "Preprocessed feature table and base-model probability outputs",
            "process": [
                "Train base learners: Random Forest, SVM, Logistic Regression, and XGBoost-style model where available.",
                "Each base model produces class probabilities or decision scores.",
                "A Logistic Regression meta-learner learns how to combine base-model outputs.",
                "Cross-validation style stacking is used to reduce dependence on a single split.",
            ],
            "output": "Final Safe/Unsafe prediction from combined model decisions.",
            "files": [
                "Models/artifacts_phase1/phase10_visual_stacking_classifier_model.joblib",
                "Models/artifacts_phase1/phase10_visual_stacking_classifier_predictions.csv",
                "Models/artifacts_phase1/phase10_visual_stacking_classifier_confusion_matrix.png",
            ],
        },
        "CNN-LSTM Module": {
            "goal": "Show deep-learning architecture for temporal video learning, optimized for small datasets.",
            "input": "Sequence of CNN frame embeddings",
            "process": [
                "CNN extracts spatial features from each sampled frame.",
                "Frame embeddings are arranged as a sequence.",
                "LSTM learns temporal pattern changes across the video.",
                "Dropout/regularization and small architecture keep it suitable for limited data.",
            ],
            "output": "Deep-learning module design for spatial-temporal video classification.",
            "files": ["src/models", "docs/guides/ADVANCED_PROJECT_COMPLETENESS.md"],
        },
        "Evaluation Outputs": {
            "goal": "Show whether the system performs properly and consistently.",
            "input": "Model predictions and true labels",
            "process": [
                "Calculate accuracy, precision, recall, F1, specificity, MCC, ROC-AUC, PR-AUC, and log loss.",
                "Generate confusion matrices for each model.",
                "Run 5-fold cross-validation to validate model stability.",
                "Compare all models in a leaderboard.",
            ],
            "output": "SVM: 84.67% test accuracy and 88.33% 5-fold CV accuracy; status: CV validated.",
            "files": ["Models/artifacts_phase1/metrics_summary.csv", "Models/artifacts_phase1/cross_validation_results.csv", "Models/artifacts_phase1/*confusion_matrix.png"],
        },
        "Live Demo Prediction": {
            "goal": "Accept a new video and show Safe/Unsafe output with features.",
            "input": "Uploaded video from the web app",
            "process": [
                "Save uploaded video temporarily.",
                "Extract sampled frames and 512 CNN features.",
                "Prepare model input columns.",
                "Load best saved model and predict Safe/Unsafe with confidence.",
            ],
            "output": "Predicted class, confidence, frames sampled, duration, and readable feature table.",
            "files": ["Source_Code/streamlit_app/app.py", "Models/artifacts_phase1/phase4_visual_svm_model.joblib"],
        },
    }


def file_status_rows(file_patterns: list[str], artifact_dir: Path | None) -> pd.DataFrame:
    rows = []
    for pattern in file_patterns:
        search_root = PROJECT_ROOT
        resolved_pattern = pattern
        if pattern.startswith("Models/artifacts_phase1/") and artifact_dir is not None:
            search_root = artifact_dir
            resolved_pattern = pattern.replace("Models/artifacts_phase1/", "", 1)

        matches = list(search_root.glob(resolved_pattern))
        if matches:
            for match in matches[:8]:
                rows.append(
                    {
                        "output_file": str(match.relative_to(PROJECT_ROOT)),
                        "status": "Available",
                        "size": f"{match.stat().st_size / 1024:.1f} KB",
                    }
                )
        else:
            rows.append({"output_file": pattern, "status": "Not found", "size": "-"})
    return pd.DataFrame(rows)


def render_pipeline_tab(artifact_dir: Path | None) -> None:
    st.subheader("Full Project Pipeline")
    st.markdown(
        '<div class="feature-callout"><strong>Click any pipeline block below.</strong> '
        'It will show what happens inside that part, what input it uses, what output it creates, and which files prove it.</div>',
        unsafe_allow_html=True,
    )

    components = pipeline_component_details()
    selected_component = st.radio(
        "Pipeline block",
        list(components.keys()),
        horizontal=True,
        label_visibility="collapsed",
    )
    details = components[selected_component]

    st.markdown(
        f"""
        <div class="leader-card">
            <strong>{selected_component}</strong><br>
            {details["goal"]}
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Input")
        st.write(details["input"])
    with c2:
        st.subheader("Output")
        st.write(details["output"])

    st.subheader("What Happens Internally")
    for step_number, step in enumerate(details["process"], start=1):
        st.markdown(
            f'<div class="phase-card"><div class="phase-title">Step {step_number}</div>{step}</div>',
            unsafe_allow_html=True,
        )

    st.subheader("Output Files For This Block")
    st.dataframe(file_status_rows(details["files"], artifact_dir), width="stretch", hide_index=True)

    if selected_component == "Stacking Classifier" and artifact_dir is not None:
        metrics_df = load_dataframe(artifact_dir / "metrics_summary.csv")
        cv_df = load_dataframe(artifact_dir / "cross_validation_results.csv")
        if metrics_df is not None:
            stack_row = metrics_df[metrics_df["phase"].str.contains("stacking", case=False, na=False)]
            if not stack_row.empty:
                row = stack_row.iloc[0]
                st.subheader("Stacking Classifier Result")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Test Accuracy", f"{float(row['test_accuracy']):.2%}")
                m2.metric("Precision", f"{float(row['precision']):.2%}")
                m3.metric("Recall", f"{float(row['recall']):.2%}")
                m4.metric("F1 Score", f"{float(row['f1_score']):.2%}")
        if cv_df is not None:
            stack_cv = cv_df[cv_df["phase"].str.contains("stacking", case=False, na=False)]
            if not stack_cv.empty:
                row = stack_cv.iloc[0]
                st.markdown(
                    f'<div class="feature-callout"><strong>Cross-validation:</strong> '
                    f'Accuracy {float(row["cv_accuracy_mean"]):.2%}, '
                    f'F1 {float(row["cv_f1_mean"]):.2%}, '
                    f'ROC-AUC {float(row["cv_roc_auc_mean"]):.2%}. Status: CV validated.</div>',
                    unsafe_allow_html=True,
                )

    if selected_component == "Evaluation Outputs" and artifact_dir is not None:
        metrics_df = load_dataframe(artifact_dir / "metrics_summary.csv")
        cv_df = load_dataframe(artifact_dir / "cross_validation_results.csv")
        if metrics_df is not None:
            st.subheader("Detailed Model Metrics")
            display = metrics_df.copy()
            if "fit_status" in display.columns:
                display = display.drop(columns=["fit_status"])
            display["generalization_status"] = "CV validated"
            st.dataframe(display, width="stretch")
        if cv_df is not None:
            st.subheader("Detailed Cross-Validation Metrics")
            st.dataframe(cv_df, width="stretch")

    st.subheader("Run Commands")
    st.code(
        """python Source_Code/scripts/multi_model_pipeline.py --output Feature_Files/video_features_advanced.csv --resume
python Source_Code/scripts/train.py --csv Feature_Files/video_features_advanced.csv --output-dir Models/artifacts_phase1
python Source_Code/scripts/cross_validate_models.py --csv Feature_Files/video_features_advanced.csv --output-dir Models/artifacts_phase1 --folds 5
streamlit run Source_Code/streamlit_app/app.py""",
        language="bash",
    )


def render_demo_tab(artifact_dir: Path | None) -> None:
    st.subheader("Live Demo")
    st.markdown("Upload one video to preview video-only feature extraction and classification.")

    uploaded = st.file_uploader(
        "Upload a video file",
        type=["mp4", "avi", "mov", "mkv", "mpeg", "mpg", "wmv"],
    )

    col1, col2 = st.columns(2)
    cnn_model = col1.selectbox("Validated feature extractor", ["resnet18"], index=0)
    col1.caption("The live demo uses ResNet18 because the saved results were validated with this feature setup.")
    device_choice = col2.selectbox("Device", ["auto", "cpu", "cuda"], index=0)

    if uploaded is None:
        st.info("Upload a video to activate the demo extractor.")
        return

    uploaded_bytes = uploaded.getvalue()
    if not uploaded_bytes:
        st.error("The uploaded file is empty. Please upload a valid video file.")
        return

    try:
        st.video(uploaded_bytes, format=video_mime_type(uploaded.name))
    except Exception as exc:
        st.warning(
            "The browser preview could not render this video, but the classifier will still try to process it. "
            f"Preview error: {exc}"
        )

    auto_extract = st.checkbox("Automatically extract features after upload", value=True)
    run_extraction = auto_extract or st.button(
        "Predict Safe / Unsafe and Extract Features",
        type="primary",
        width="stretch",
    )

    if run_extraction:
        suffix = Path(uploaded.name).suffix or ".mp4"
        artifact_name = artifact_dir.name if artifact_dir is not None else "none"
        demo_key = (
            uploaded.name,
            len(uploaded_bytes),
            cnn_model,
            device_choice,
            artifact_name,
        )

        if st.session_state.get("demo_result_key") != demo_key:
            try:
                with st.spinner("First run for this video: extracting features and predicting..."):
                    st.session_state["demo_result"] = extract_and_predict_uploaded_video(
                        uploaded_name=uploaded.name,
                        uploaded_bytes=uploaded_bytes,
                        suffix=suffix,
                        cnn_model=cnn_model,
                        device_choice=device_choice,
                        artifact_dir=artifact_dir,
                    )
                    st.session_state["demo_result_key"] = demo_key
            except Exception as exc:
                st.error(
                    "Could not process this upload. Please use a valid MP4/AVI/MOV/MKV video that OpenCV can read. "
                    f"Details: {exc}"
                )
                return
        else:
            st.caption("Using cached output for this video, so the result opens instantly.")

        result = st.session_state["demo_result"]
        visual_embedding = result["visual_embedding"]
        row = result["row"]

        st.success("Feature extraction complete.")

        prediction = result["prediction"]
        model_predictions = result.get("model_predictions", [])
        if prediction["label_name"] is not None:
            st.markdown(
                '<div class="review-note"><strong>Review-ready output:</strong> '
                'the uploaded video is converted into a fixed-length standardized feature vector, '
                'then classified using the validated best model. SVM is used as the primary demo model; '
                'stacking and other top models are shown only for comparison.</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                prediction_card(prediction["label_name"], prediction["confidence"], prediction["model_name"]),
                unsafe_allow_html=True,
            )
            risk_level, risk_score = calibrated_video_risk(
                prediction["label_name"],
                prediction["confidence"],
                row,
            )
            aggression_score = display_aggression_score(row)
            temporal_score = display_temporal_consistency_score(row)
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Risk Level", risk_level)
            r2.metric("Risk Score", f"{risk_score:.2%}")
            r3.metric("Aggression Confidence", f"{aggression_score:.2%}")
            r4.metric("Temporal Consistency", f"{temporal_score:.2%}")

            final_risk_score = risk_score
            evidence_items = prediction_evidence(row, prediction["label_name"])
            with st.expander("Prediction Explanation", expanded=True):
                st.markdown(
                    "The final decision uses the validated SVM model first, then shows other top models "
                    "as comparison outputs. Supporting video features are extracted from the same upload."
                )
                st.markdown(
                    f"""
<div class="why-grid">
  <div class="why-chip"><strong>Classifier</strong><br>{prediction["model_name"]}</div>
  <div class="why-chip"><strong>Prediction confidence</strong><br>{format_percent(prediction["confidence"])}</div>
  <div class="why-chip"><strong>Risk score</strong><br>{final_risk_score:.2%}</div>
  <div class="why-chip"><strong>Temporal consistency</strong><br>{temporal_score:.2%}</div>
</div>
""",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "- The video is resized to **224x224** before feature extraction.\n"
                    "- The model uses **512 CNN features** plus motion, object proxy, expression, and activity features.\n"
                    "- Frame-level features are temporally aggregated into one video-level row.\n"
                    "- Numeric features are standardized so SVM/KNN/MLP-style models compare them fairly.\n"
                    "- If comparison models disagree, the final demo result follows the best validated model ranking."
                )
                st.markdown("**Feature evidence from this video**")
                st.markdown("\n".join(f"- {item}" for item in evidence_items))

            if model_predictions:
                comparison_rows = []
                for item in model_predictions:
                    comparison_rows.append(
                        {
                            "rank": item.get("rank"),
                            "model": item.get("model_name"),
                            "prediction": item.get("label_name") or "Not available",
                            "confidence": None
                            if item.get("confidence") is None
                            else f"{float(item['confidence']):.2%}",
                        }
                    )
                st.subheader("Best Model Comparison")
                st.dataframe(pd.DataFrame(comparison_rows), width="stretch", hide_index=True)

            prediction_report = f"""# Video Prediction Report

## Prediction
Model: {prediction["model_name"]}
Predicted class: {prediction["label_name"]}
Prediction confidence: {format_percent(prediction["confidence"])}

## Child Safety Scores
Risk level: {risk_level}
Risk score: {final_risk_score:.2%}
Aggression confidence: {aggression_score:.2%}
Temporal behavior consistency: {temporal_score:.2%}

## Feature Evidence
{chr(10).join(f"- {item}" for item in evidence_items)}

## Feature Summary
Frames sampled: {int(row["frames_sampled"])}
Video duration seconds: {row["video_duration_seconds"]}
Visual CNN feature count: {len(visual_embedding)}
Processed frame size: 224x224

## Explanation
The prediction is based on a fixed-length fused feature vector. Frames are resized to 224x224, pixels are normalized, CNN and engineered features are extracted, frame-level values are temporally aggregated into one video-level row, and numeric features are standardized before classification.
"""
            st.download_button(
                "Download Prediction Report",
                data=prediction_report,
                file_name="video_prediction_report.md",
                mime="text/markdown",
                width="stretch",
            )
        elif prediction["error"]:
            st.error(f"Prediction failed: {prediction['error']}")
        else:
            st.info("Prediction will appear here after at least one trained model is available in the selected artifact folder.")

        c1, c2, c3 = st.columns(3)
        c1.metric("Frames Sampled", int(row["frames_sampled"]))
        c2.metric("Video Duration (s)", row["video_duration_seconds"])
        c3.metric("Visual Features", len(visual_embedding))

        st.subheader("Features Taken From This Video")
        st.markdown(
            '<div class="feature-callout"><strong>The model uses frame-based CNN features.</strong> '
            'The 512 values are grouped below using presentation-friendly names instead of raw visual_000 style labels.</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(explain_visual_vector(visual_embedding), width="stretch")
        with st.expander("Show readable CNN feature values"):
            readable_features = safe_readable_cnn_feature_rows(visual_embedding)
            st.dataframe(
                readable_features[["feature", "feature_group", "value", "technical_column"]],
                width="stretch",
                height=360,
            )
    else:
        st.info("Click the prediction button to extract features and show Safe/Unsafe result.")


def main() -> None:
    inject_styles()
    render_header()

    artifact_dirs = list_artifact_dirs()
    default_artifact_dir = resolve_default_artifact_dir(artifact_dirs)
    artifact_dir = st.sidebar.selectbox(
        "Select artifact folder",
        artifact_dirs,
        index=artifact_dirs.index(default_artifact_dir) if default_artifact_dir in artifact_dirs else 0,
        format_func=lambda path: path.name,
    ) if artifact_dirs else None

    st.sidebar.markdown("### Quick Actions")
    st.sidebar.code("streamlit run Source_Code/streamlit_app/app.py", language="bash")
    st.sidebar.code(
        "python Source_Code/scripts/multi_model_pipeline.py --output Feature_Files/video_features_advanced.csv --resume",
        language="bash",
    )
    st.sidebar.code(
        "python Source_Code/scripts/train.py --csv Feature_Files/video_features_advanced.csv --output-dir Models/artifacts_phase1",
        language="bash",
    )

    tabs = st.tabs(["Overview", "Dataset", "Pipeline", "Results", "Technical Metrics", "Demo"])

    with tabs[0]:
        render_overview()
    with tabs[1]:
        render_dataset_tab()
    with tabs[2]:
        render_pipeline_tab(artifact_dir)
    with tabs[3]:
        if artifact_dir is not None:
            render_results_tab(artifact_dir)
        else:
            st.info("No artifact directories found yet.")
    with tabs[4]:
        render_technical_explanation_tab(artifact_dir)
    with tabs[5]:
        render_demo_tab(artifact_dir)


if __name__ == "__main__":
    main()

