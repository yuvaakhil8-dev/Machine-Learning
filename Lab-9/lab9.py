
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "video_features_yolo.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "lab9"
RANDOM_STATE = 42


def load_project_dataset() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATASET_PATH)
    features = df.drop(columns=["label"])
    targets = df["label"]
    return features, targets


def base_classifiers() -> list[tuple[str, object]]:
    return [
        ("logistic_regression", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ("random_forest", RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE)),
        ("support_vector_machine", SVC(probability=True, random_state=RANDOM_STATE)),
        ("knn", KNeighborsClassifier(n_neighbors=7)),
    ]


def final_estimators() -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(n_estimators=120, random_state=RANDOM_STATE),
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def build_stacking_pipeline(final_estimator: object) -> Pipeline:
    stacking_model = StackingClassifier(
        estimators=base_classifiers(),
        final_estimator=final_estimator,
        cv=5,
        stack_method="predict_proba",
        passthrough=True,
        n_jobs=-1,
    )
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("stacking_classifier", stacking_model),
        ]
    )


def evaluate_models(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    rows = []
    trained_pipelines: dict[str, Pipeline] = {}

    for estimator_name, estimator in final_estimators().items():
        pipeline = build_stacking_pipeline(estimator)
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        trained_pipelines[estimator_name] = pipeline
        rows.append(
            {
                "final_estimator": estimator_name,
                "accuracy": accuracy,
                "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
            }
        )

    results = pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)
    return results, trained_pipelines


def explain_with_lime(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    sample_index: int,
    output_dir: Path,
) -> pd.DataFrame:
    def predict_with_feature_names(data: np.ndarray) -> np.ndarray:
        data_frame = pd.DataFrame(data, columns=x_train.columns)
        return pipeline.predict_proba(data_frame)

    explainer = LimeTabularExplainer(
        training_data=x_train.to_numpy(),
        feature_names=x_train.columns.tolist(),
        class_names=["Safe", "Unsafe"],
        mode="classification",
        discretize_continuous=True,
        random_state=RANDOM_STATE,
    )
    explanation = explainer.explain_instance(
        data_row=x_test.iloc[sample_index].to_numpy(),
        predict_fn=predict_with_feature_names,
        num_features=len(x_train.columns),
    )
    explanation.save_to_file(output_dir / "lime_pipeline_explanation.html")
    rows = [{"feature_rule": rule, "contribution": weight} for rule, weight in explanation.as_list()]
    return pd.DataFrame(rows)


def save_text_report(
    output_path: Path,
    best_name: str,
    y_test: pd.Series,
    predictions: np.ndarray,
    model_results: pd.DataFrame,
    lime_results: pd.DataFrame,
) -> None:
    report_lines = [
        "Lab 09 Results",
        "",
        "Stacking classifier comparison:",
        model_results.to_string(index=False),
        "",
        f"Best final estimator: {best_name}",
        "",
        "Classification report for best pipeline:",
        classification_report(y_test, predictions, target_names=["Safe", "Unsafe"]),
        "",
        "LIME explanation for one test instance:",
        lime_results.to_string(index=False),
    ]
    output_path.write_text("\n".join(report_lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    features, targets = load_project_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        targets,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=targets,
    )

    print("\nA1 stacking classifier with different meta-models")
    model_results, trained_pipelines = evaluate_models(x_train, x_test, y_train, y_test)
    print(model_results.to_string(index=False))

    best_name = model_results.iloc[0]["final_estimator"]
    best_pipeline = trained_pipelines[best_name]
    best_predictions = best_pipeline.predict(x_test)

    print("\nA2 pipeline execution")
    print("Pipeline steps:", [name for name, _ in best_pipeline.steps])
    print("Best pipeline accuracy:", round(accuracy_score(y_test, best_predictions), 4))
    print(classification_report(y_test, best_predictions, target_names=["Safe", "Unsafe"]))

    print("\nA3 LIME explanation for the best pipeline")
    sample_index = 0
    lime_results = explain_with_lime(best_pipeline, x_train, x_test, sample_index, OUTPUT_DIR)
    print(lime_results.to_string(index=False))

    model_results.to_csv(OUTPUT_DIR / "stacking_results.csv", index=False)
    lime_results.to_csv(OUTPUT_DIR / "lime_explanation.csv", index=False)
    save_text_report(
        OUTPUT_DIR / "lab9_report.txt",
        best_name,
        y_test,
        best_predictions,
        model_results,
        lime_results,
    )
    print(f"\nSaved Lab 9 results to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
