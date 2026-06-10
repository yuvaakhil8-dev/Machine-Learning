from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


EXCLUDED_COLUMNS = {"label", "video_name", "video_path", "class_name"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run k-fold cross-validation for the video classification models.")
    parser.add_argument("--csv", default="Feature_Files/video_features_advanced.csv", help="Feature CSV containing label and visual feature columns.")
    parser.add_argument("--output-dir", default="Models/artifacts_phase1", help="Folder where CV outputs will be saved.")
    parser.add_argument("--folds", type=int, default=5, help="Number of stratified folds.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in df.columns
        if column not in EXCLUDED_COLUMNS and pd.api.types.is_numeric_dtype(df[column])
    ]


def preprocessor(columns: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                columns,
            )
        ],
        remainder="drop",
    )


def xgboost_model(random_state: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=180,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=1,
    )


def model_suite(random_state: int, columns: list[str]) -> dict[str, Pipeline]:
    prep = preprocessor(columns)
    logistic = LogisticRegression(max_iter=2500, random_state=random_state)
    decision_tree = DecisionTreeClassifier(
        max_depth=8,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=random_state,
    )
    random_forest = RandomForestClassifier(
        n_estimators=220,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    svm = SVC(
        kernel="rbf",
        C=2.0,
        gamma="scale",
        probability=True,
        class_weight="balanced",
        random_state=random_state,
    )
    knn = KNeighborsClassifier(n_neighbors=7)
    naive_bayes = GaussianNB()
    adaboost = AdaBoostClassifier(n_estimators=150, learning_rate=0.5, random_state=random_state)
    xgb = xgboost_model(random_state)
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        alpha=0.001,
        learning_rate_init=0.001,
        early_stopping=True,
        validation_fraction=0.15,
        max_iter=600,
        random_state=random_state,
    )
    stacking = StackingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=180, min_samples_leaf=2, class_weight="balanced", random_state=random_state, n_jobs=-1)),
            ("svm", SVC(kernel="rbf", C=2.0, gamma="scale", probability=True, class_weight="balanced", random_state=random_state)),
            ("xgb", xgboost_model(random_state)),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    alpha=0.001,
                    learning_rate_init=0.001,
                    early_stopping=True,
                    validation_fraction=0.15,
                    max_iter=600,
                    random_state=random_state,
                ),
            ),
        ],
        final_estimator=LogisticRegression(max_iter=2500, random_state=random_state),
        stack_method="predict_proba",
        cv=3,
        n_jobs=1,
    )

    return {
        "phase1_visual_logistic": Pipeline(steps=[("preprocessor", prep), ("model", logistic)]),
        "phase2_visual_decision_tree": Pipeline(steps=[("preprocessor", prep), ("model", decision_tree)]),
        "phase3_visual_random_forest": Pipeline(steps=[("preprocessor", prep), ("model", random_forest)]),
        "phase4_visual_svm": Pipeline(steps=[("preprocessor", prep), ("model", svm)]),
        "phase5_visual_knn": Pipeline(steps=[("preprocessor", prep), ("model", knn)]),
        "phase6_visual_naive_bayes": Pipeline(steps=[("preprocessor", prep), ("model", naive_bayes)]),
        "phase7_visual_adaboost": Pipeline(steps=[("preprocessor", prep), ("model", adaboost)]),
        "phase8_visual_xgboost": Pipeline(steps=[("preprocessor", prep), ("model", xgb)]),
        "phase9_visual_mlp_classifier": Pipeline(steps=[("preprocessor", prep), ("model", mlp)]),
        "phase10_visual_stacking_classifier": Pipeline(steps=[("preprocessor", prep), ("model", stacking)]),
    }


def summarize_scores(name: str, scores: dict[str, object], folds: int) -> dict[str, float | str | int]:
    row: dict[str, float | str | int] = {"phase": name, "folds": folds}
    for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        values = scores[f"test_{metric}"]
        row[f"cv_{metric}_mean"] = round(float(values.mean()), 4)
        row[f"cv_{metric}_std"] = round(float(values.std()), 4)
    row["fit_time_mean_seconds"] = round(float(scores["fit_time"].mean()), 2)
    return row


def save_chart(results: pd.DataFrame, output_dir: Path) -> None:
    chart_df = results.sort_values("cv_accuracy_mean", ascending=True)
    labels = chart_df["phase"].str.replace("_", " ").str.title()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(labels, chart_df["cv_accuracy_mean"], xerr=chart_df["cv_accuracy_std"], color="#2563eb", alpha=0.88)
    ax.set_xlabel("Cross-validation accuracy")
    ax.set_title("5-Fold Cross-Validation Accuracy")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_dir / "cross_validation_accuracy.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    if "label" not in df.columns:
        raise ValueError("Input CSV must contain a label column.")

    columns = numeric_feature_columns(df)
    X = df.drop(columns=["label"])
    y = df["label"]

    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.random_state)
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "roc_auc": "roc_auc",
    }

    rows = []
    for name, model in model_suite(args.random_state, columns).items():
        print(f"Running {args.folds}-fold CV for {name}...")
        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
            return_train_score=False,
        )
        rows.append(summarize_scores(name, scores, args.folds))

    results = pd.DataFrame(rows).sort_values("cv_accuracy_mean", ascending=False)
    results.to_csv(output_dir / "cross_validation_results.csv", index=False)
    save_chart(results, output_dir)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
