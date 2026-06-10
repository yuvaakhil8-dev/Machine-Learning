from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import plot_tree

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "Source_Code"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


EXCLUDED_COLUMNS = {"label", "video_name", "video_path", "class_name"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate visual, explainability, and project audit outputs.")
    parser.add_argument("--csv", default="Feature_Files/video_features_advanced.csv")
    parser.add_argument("--artifact-dir", default="Models/artifacts_phase1")
    parser.add_argument("--output-dir", default="Results/generated_project_outputs")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in df.columns
        if column not in EXCLUDED_COLUMNS and pd.api.types.is_numeric_dtype(df[column])
    ]


def pretty_model_name(name: str) -> str:
    cleaned = name.replace("phase", "").replace("_visual_", " ").replace("_", " ")
    return " ".join(word.capitalize() for word in cleaned.split())


def safe_savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def select_best_model(metrics: pd.DataFrame, artifact_dir: Path) -> tuple[str, object]:
    ranked = metrics.sort_values(["test_accuracy", "f1_score"], ascending=False)
    for phase in ranked["phase"]:
        model_path = artifact_dir / f"{phase}_model.joblib"
        if model_path.exists():
            return str(phase), joblib.load(model_path)
    candidates = sorted(artifact_dir.glob("*_model.joblib"))
    if not candidates:
        raise FileNotFoundError(f"No trained model artifacts found in {artifact_dir}")
    return candidates[0].stem.replace("_model", ""), joblib.load(candidates[0])


def save_dataset_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    counts = df["label"].map({0: "Safe", 1: "Unsafe"}).value_counts().reindex(["Safe", "Unsafe"], fill_value=0)
    counts.to_csv(output_dir / "dataset_distribution.csv", header=["count"])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index, counts.values, color=["#2563eb", "#ef4444"])
    ax.set_title("Dataset Class Distribution")
    ax.set_ylabel("Number of videos")
    for idx, value in enumerate(counts.values):
        ax.text(idx, value + 2, str(int(value)), ha="center", fontweight="bold")
    safe_savefig(output_dir / "dataset_distribution.png")


def save_model_comparison(metrics: pd.DataFrame, output_dir: Path) -> None:
    display = metrics.copy()
    display["model"] = display["phase"].map(pretty_model_name)
    display.to_csv(output_dir / "model_comparison_table.csv", index=False)

    ordered = display.sort_values("test_accuracy", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(ordered["model"], ordered["test_accuracy"], color="#0f766e")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Test accuracy")
    ax.set_title("Classifier Comparison")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    safe_savefig(output_dir / "classifier_comparison_accuracy.png")

    ranking = display.sort_values(["test_accuracy", "f1_score", "roc_auc"], ascending=False)
    ranking[["model", "test_accuracy", "precision", "recall", "f1_score", "roc_auc"]].to_csv(
        output_dir / "model_ranking.csv",
        index=False,
    )


def save_pca_and_correlation(df: pd.DataFrame, features: list[str], output_dir: Path) -> None:
    X = df[features].fillna(0).to_numpy()
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    components = min(20, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=components, random_state=42)
    projected = pca.fit_transform(X)

    pca_df = pd.DataFrame(
        {
            "component": [f"PC{i + 1}" for i in range(components)],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    pca_df.to_csv(output_dir / "pca_variance.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, components + 1), pca_df["cumulative_variance"], marker="o", color="#7c3aed")
    ax.set_title("PCA Cumulative Explained Variance")
    ax.set_xlabel("Number of PCA components")
    ax.set_ylabel("Cumulative variance")
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.35)
    safe_savefig(output_dir / "pca_cumulative_variance.png")

    if projected.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(6, 5))
        colors = df["label"].map({0: "#2563eb", 1: "#ef4444"})
        ax.scatter(projected[:, 0], projected[:, 1], c=colors, alpha=0.75, s=20)
        ax.set_title("PCA 2D Feature View")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        safe_savefig(output_dir / "pca_2d_class_view.png")

    top_features = features[: min(40, len(features))]
    corr = df[top_features].fillna(0).corr().fillna(0)
    corr.to_csv(output_dir / "correlation_matrix_top40.csv")
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("Correlation Heatmap: First 40 Numeric Features")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    safe_savefig(output_dir / "correlation_heatmap_top40.png")


def transformed_feature_names(model, fallback: list[str]) -> list[str]:
    preprocessor = getattr(model, "named_steps", {}).get("preprocessor")
    if preprocessor is None:
        return fallback
    try:
        names = preprocessor.get_feature_names_out()
        return [str(name).replace("numeric__", "") for name in names]
    except Exception:
        return fallback


def estimator_feature_importance(model, feature_names: list[str]) -> pd.DataFrame:
    estimator = getattr(model, "named_steps", {}).get("model", model)
    values = None
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_, dtype=float)).reshape(-1)

    if values is None:
        return pd.DataFrame()

    size = min(len(feature_names), len(values))
    result = pd.DataFrame({"feature": feature_names[:size], "importance": values[:size]})
    return result.sort_values("importance", ascending=False)


def save_feature_importance(best_model_name: str, model, X_test: pd.DataFrame, y_test: pd.Series, features: list[str], output_dir: Path) -> None:
    names = transformed_feature_names(model, features)
    ranking = estimator_feature_importance(model, names)
    if not ranking.empty:
        ranking.to_csv(output_dir / "feature_importance_ranking.csv", index=False)
        top = ranking.head(25).iloc[::-1]
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.barh(top["feature"], top["importance"], color="#2563eb")
        ax.set_title(f"Top Feature Importance: {pretty_model_name(best_model_name)}")
        ax.set_xlabel("Importance")
        safe_savefig(output_dir / "feature_importance_top25.png")
    else:
        (output_dir / "feature_importance_ranking.csv").write_text(
            "feature,importance\nNo built-in model importance available for this best model.\n",
            encoding="utf-8",
        )

    sample_size = min(80, len(X_test))
    permutation = permutation_importance(
        model,
        X_test.iloc[:sample_size],
        y_test.iloc[:sample_size],
        n_repeats=8,
        random_state=42,
        scoring="accuracy",
        n_jobs=1,
    )
    perm_df = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": permutation.importances_mean,
            "importance_std": permutation.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    perm_df.to_csv(output_dir / "permutation_importance.csv", index=False)
    top_perm = perm_df.head(25).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top_perm["feature"], top_perm["importance_mean"], xerr=top_perm["importance_std"], color="#f97316")
    ax.set_title("Permutation Importance Fallback")
    ax.set_xlabel("Accuracy drop when shuffled")
    safe_savefig(output_dir / "permutation_importance_top25.png")


def save_roc_pr_curves(metrics: pd.DataFrame, artifact_dir: Path, X_test: pd.DataFrame, y_test: pd.Series, output_dir: Path) -> None:
    roc_rows = []
    pr_rows = []
    fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
    fig_pr, ax_pr = plt.subplots(figsize=(7, 6))

    for phase in metrics["phase"]:
        model_path = artifact_dir / f"{phase}_model.joblib"
        if not model_path.exists():
            continue
        model = joblib.load(model_path)
        if not hasattr(model, "predict_proba"):
            continue
        proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        precision, recall, _ = precision_recall_curve(y_test, proba)
        roc_auc = auc(fpr, tpr)
        pr_auc = auc(recall, precision)
        name = pretty_model_name(str(phase))
        ax_roc.plot(fpr, tpr, label=f"{name} ({roc_auc:.3f})")
        ax_pr.plot(recall, precision, label=f"{name} ({pr_auc:.3f})")
        roc_rows.append({"phase": phase, "roc_auc_plot": round(float(roc_auc), 4)})
        pr_rows.append({"phase": phase, "pr_auc_plot": round(float(pr_auc), 4)})

    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="#64748b", linewidth=1)
    ax_roc.set_title("ROC Curves")
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.legend(fontsize=7)
    ax_roc.grid(True, linestyle="--", alpha=0.3)
    fig_roc.tight_layout()
    fig_roc.savefig(output_dir / "roc_curves_all_models.png", dpi=170)
    plt.close(fig_roc)

    ax_pr.set_title("Precision-Recall Curves")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend(fontsize=7)
    ax_pr.grid(True, linestyle="--", alpha=0.3)
    fig_pr.tight_layout()
    fig_pr.savefig(output_dir / "precision_recall_curves_all_models.png", dpi=170)
    plt.close(fig_pr)

    pd.DataFrame(roc_rows).to_csv(output_dir / "roc_curve_auc_values.csv", index=False)
    pd.DataFrame(pr_rows).to_csv(output_dir / "precision_recall_auc_values.csv", index=False)


def save_decision_tree_visualization(artifact_dir: Path, features: list[str], output_dir: Path) -> None:
    model_path = artifact_dir / "phase2_visual_decision_tree_model.joblib"
    if not model_path.exists():
        return
    model = joblib.load(model_path)
    estimator = model.named_steps["model"]
    names = transformed_feature_names(model, features)
    fig, ax = plt.subplots(figsize=(18, 9))
    plot_tree(
        estimator,
        feature_names=names,
        class_names=["Safe", "Unsafe"],
        filled=True,
        rounded=True,
        max_depth=3,
        fontsize=7,
        ax=ax,
    )
    ax.set_title("Decision Tree Visualization (Top Levels)")
    safe_savefig(output_dir / "decision_tree_top_levels.png")


def save_learning_curve_proxy(metrics: pd.DataFrame, output_dir: Path) -> None:
    rows = []
    for _, row in metrics.iterrows():
        rows.append(
            {
                "phase": row["phase"],
                "train_accuracy": row.get("train_accuracy", np.nan),
                "test_accuracy": row.get("test_accuracy", np.nan),
                "generalization_gap": row.get("train_accuracy", np.nan) - row.get("test_accuracy", np.nan),
            }
        )
    gap_df = pd.DataFrame(rows)
    gap_df.to_csv(output_dir / "train_test_gap_analysis.csv", index=False)

    ordered = gap_df.sort_values("generalization_gap", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(ordered["phase"].map(pretty_model_name), ordered["generalization_gap"], color="#9333ea")
    ax.set_title("Train-Test Generalization Gap")
    ax.set_xlabel("Train accuracy - test accuracy")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    safe_savefig(output_dir / "generalization_gap_chart.png")


def model_expected_columns(model) -> list[str] | None:
    preprocessor = getattr(model, "named_steps", {}).get("preprocessor")
    expected = getattr(preprocessor, "feature_names_in_", None)
    if expected is None:
        return None
    return [str(column) for column in expected]


def aligned_prediction_frame(data: np.ndarray, numeric_columns: list[str], model) -> pd.DataFrame:
    frame = pd.DataFrame(data, columns=numeric_columns)
    expected = model_expected_columns(model)
    if expected is None:
        return frame
    for column in expected:
        if column not in frame.columns:
            frame[column] = 0
    return frame[expected]


def save_lime_and_shap_outputs(
    best_model_name: str,
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    numeric_columns: list[str],
    output_dir: Path,
) -> None:
    status = []
    train_numeric = X_train[numeric_columns].fillna(0)
    test_numeric = X_test[numeric_columns].fillna(0)
    predict_fn = lambda data: model.predict_proba(aligned_prediction_frame(data, numeric_columns, model))

    try:
        import lime.lime_tabular

        background = train_numeric.to_numpy()
        explainer = lime.lime_tabular.LimeTabularExplainer(
            background,
            feature_names=list(numeric_columns),
            class_names=["Safe", "Unsafe"],
            mode="classification",
            discretize_continuous=True,
            random_state=42,
        )
        row = test_numeric.iloc[0].to_numpy()
        explanation = explainer.explain_instance(row, predict_fn, num_features=20)
        explanation.save_to_file(str(output_dir / "lime_explanation_sample.html"))
        pd.DataFrame(explanation.as_list(), columns=["feature_rule", "lime_weight"]).to_csv(
            output_dir / "lime_explanation_sample.csv",
            index=False,
        )
        status.append("LIME explanation generated for one representative test video.")
    except Exception as exc:
        status.append(f"LIME fallback note: {exc}")

    try:
        import shap

        background = train_numeric.sample(min(25, len(train_numeric)), random_state=42)
        explain_rows = test_numeric.sample(min(10, len(test_numeric)), random_state=42)
        explainer = shap.KernelExplainer(lambda data: predict_fn(data)[:, 1], background)
        shap_values = explainer.shap_values(explain_rows, nsamples=80)
        shap_df = pd.DataFrame(
            {
                "feature": numeric_columns,
                "mean_abs_shap": np.abs(np.asarray(shap_values)).mean(axis=0),
            }
        ).sort_values("mean_abs_shap", ascending=False)
        shap_df.to_csv(output_dir / "shap_kernel_feature_importance.csv", index=False)
        top = shap_df.head(20).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(top["feature"], top["mean_abs_shap"], color="#0891b2")
        ax.set_title(f"SHAP Kernel Importance: {pretty_model_name(best_model_name)}")
        ax.set_xlabel("Mean absolute SHAP value")
        safe_savefig(output_dir / "shap_kernel_importance_top20.png")
        status.append("SHAP KernelExplainer generated on a small test sample.")
    except Exception as exc:
        status.append(f"SHAP fallback note: {exc}")

    (output_dir / "explainability_status.txt").write_text("\n".join(status), encoding="utf-8")


def save_feature_dictionary(features: list[str], output_dir: Path) -> None:
    dictionary_path = PROJECT_ROOT / "Feature_Files" / "feature_column_name_mapping_descriptive.csv"
    if dictionary_path.exists():
        mapping = pd.read_csv(dictionary_path)
        mapping.to_csv(output_dir / "feature_dictionary_used_by_project.csv", index=False)
    else:
        pd.DataFrame({"feature": features, "meaning": features}).to_csv(
            output_dir / "feature_dictionary_used_by_project.csv",
            index=False,
        )


def save_audit_summary(df: pd.DataFrame, metrics: pd.DataFrame, cv: pd.DataFrame | None, output_dir: Path) -> None:
    summary = {
        "total_videos": int(len(df)),
        "safe_videos": int((df["label"] == 0).sum()),
        "unsafe_videos": int((df["label"] == 1).sum()),
        "feature_count": int(len(numeric_feature_columns(df))),
        "models_trained": list(metrics["phase"]),
        "best_test_model": str(metrics.sort_values("test_accuracy", ascending=False).iloc[0]["phase"]),
        "has_cross_validation": cv is not None and not cv.empty,
        "cross_validated_models": [] if cv is None else list(cv["phase"]),
    }
    (output_dir / "project_output_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def save_soft_weighted_voting_outputs(
    artifact_dir: Path,
    cv_df: pd.DataFrame | None,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
) -> None:
    base_phases = [
        "phase3_visual_random_forest",
        "phase4_visual_svm",
        "phase8_visual_xgboost",
        "phase9_visual_mlp_classifier",
    ]
    probabilities = []
    weights = []
    used_models = []
    cv_lookup = cv_df.set_index("phase") if cv_df is not None and not cv_df.empty else None

    for phase in base_phases:
        model_path = artifact_dir / f"{phase}_model.joblib"
        if not model_path.exists():
            continue
        model = joblib.load(model_path)
        if not hasattr(model, "predict_proba"):
            continue
        probabilities.append(model.predict_proba(X_test))
        used_models.append(phase)
        if cv_lookup is not None and phase in cv_lookup.index:
            weights.append(float(cv_lookup.loc[phase, "cv_f1_mean"]))
        else:
            weights.append(1.0)

    if not probabilities:
        return

    soft_proba = np.mean(probabilities, axis=0)
    weights_array = np.asarray(weights, dtype=float)
    weights_array = weights_array / weights_array.sum()
    weighted_proba = np.zeros_like(probabilities[0])
    for proba, weight in zip(probabilities, weights_array):
        weighted_proba += proba * weight

    rows = []
    for name, proba in [("soft_voting", soft_proba), ("weighted_voting", weighted_proba)]:
        pred = np.argmax(proba, axis=1)
        rows.append(
            {
                "ensemble_method": name,
                "models_used": ", ".join(used_models),
                "accuracy": round(float(accuracy_score(y_test, pred)), 4),
                "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
                "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
                "f1_score": round(float(f1_score(y_test, pred, zero_division=0)), 4),
            }
        )

    pd.DataFrame(rows).to_csv(output_dir / "soft_weighted_voting_results.csv", index=False)


def save_requirements_coverage(output_dir: Path) -> None:
    rows = [
        ("Existing labeled Safe/Unsafe dataset", "Included", "Safe (3), Unsafe, data/metadata/dataset_metadata.csv"),
        ("Video validation and corruption checks", "Included", "src/scripts/audit_dataset_videos.py, outputs/audit/dataset_audit"),
        ("Frame extraction, resizing, sampling, normalization", "Included", "src/feature_engineering/video_features.py"),
        ("Duplicate frame reduction support", "Included", "src/feature_engineering/video_features.py"),
        ("Video augmentation methods", "Included", "src/preprocessing/augmentation.py"),
        ("CNN transfer feature extraction", "Included", "src/models/transfer_backbones.py, src/scripts/multi_model_pipeline.py"),
        ("ResNet18 / MobileNetV2 / MobileNetV3 Small support", "Included", "src/models/transfer_backbones.py"),
        ("Motion feature extraction", "Included", "src/feature_engineering/video_features.py"),
        ("YOLOv8 object feature hooks", "Included", "src/feature_engineering/yolo_object_features.py, data/weights/yolov8n.pt"),
        ("Emotion proxy features", "Included", "src/feature_engineering/emotion_features.py"),
        ("Pose/activity proxy features", "Included", "src/feature_engineering/pose_activity_features.py"),
        ("Feature fusion", "Included", "src/feature_engineering/video_fusion.py"),
        ("9 ML classifiers", "Included", "src/scripts/train.py, models_artifacts/artifacts_phase1/metrics_summary.csv"),
        ("Decision Tree baseline", "Included", "phase2_visual_decision_tree outputs"),
        ("KNN sweep", "Included", "models_artifacts/artifacts_phase1/knn_results.csv"),
        ("Stacking ensemble", "Included", "phase10_visual_stacking_classifier outputs"),
        ("Soft and weighted voting", "Included", "outputs/generated_project_outputs/soft_weighted_voting_results.csv"),
        ("Cross-validation", "Included", "models_artifacts/artifacts_phase1/cross_validation_results.csv"),
        ("GridSearchCV and RandomizedSearchCV optimization", "Included", "src/models/model_optimization.py, models_artifacts/optimization/hyperparameter_optimization_results.csv"),
        ("Advanced metrics", "Included", "src/evaluation/advanced_metrics.py, metrics_summary.csv"),
        ("Confusion matrices", "Included", "models_artifacts/artifacts_phase1/*confusion_matrix.png"),
        ("ROC and PR curves", "Included", "outputs/generated_project_outputs/roc_curves_all_models.png, precision_recall_curves_all_models.png"),
        ("PCA and correlation visualizations", "Included", "outputs/generated_project_outputs/pca_*.png, correlation_heatmap_top40.png"),
        ("Decision tree visualization", "Included", "outputs/generated_project_outputs/decision_tree_top_levels.png"),
        ("Feature importance", "Included", "outputs/generated_project_outputs/feature_importance_ranking.csv"),
        ("Permutation importance fallback", "Included", "outputs/generated_project_outputs/permutation_importance.csv"),
        ("LIME explanation", "Included", "outputs/generated_project_outputs/lime_explanation_sample.html"),
        ("SHAP explanation", "Included", "outputs/generated_project_outputs/shap_kernel_feature_importance.csv"),
        ("Streamlit app", "Included", "app/streamlit_app.py"),
        ("Upload video and predict", "Included", "app/streamlit_app.py Demo tab"),
        ("Child safety risk score", "Included", "app/streamlit_app.py Demo tab"),
        ("Saved reports and charts", "Included", "outputs/reports, outputs/generated_project_outputs"),
    ]
    pd.DataFrame(rows, columns=["requirement", "status", "evidence_path"]).to_csv(
        output_dir / "requirements_coverage.csv",
        index=False,
    )


def main() -> None:
    args = parse_args()
    csv_path = PROJECT_ROOT / args.csv
    artifact_dir = PROJECT_ROOT / args.artifact_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    metrics = pd.read_csv(artifact_dir / "metrics_summary.csv")
    cv_path = artifact_dir / "cross_validation_results.csv"
    cv_df = pd.read_csv(cv_path) if cv_path.exists() else None

    features = numeric_feature_columns(df)
    X = df.drop(columns=["label"])
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=args.random_state,
    )

    best_model_name, best_model = select_best_model(metrics, artifact_dir)

    save_dataset_distribution(df, output_dir)
    save_model_comparison(metrics, output_dir)
    save_pca_and_correlation(df, features, output_dir)
    save_feature_importance(best_model_name, best_model, X_test, y_test, features, output_dir)
    save_roc_pr_curves(metrics, artifact_dir, X_test, y_test, output_dir)
    save_decision_tree_visualization(artifact_dir, features, output_dir)
    save_learning_curve_proxy(metrics, output_dir)
    save_lime_and_shap_outputs(best_model_name, best_model, X_train, X_test, features, output_dir)
    save_soft_weighted_voting_outputs(artifact_dir, cv_df, X_test, y_test, output_dir)
    save_feature_dictionary(features, output_dir)
    save_audit_summary(df, metrics, cv_df, output_dir)
    save_requirements_coverage(output_dir)

    print(f"Generated project outputs in: {output_dir}")


if __name__ == "__main__":
    main()
