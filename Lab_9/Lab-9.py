from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

from lab_experiments.common import ensure_output_dir, load_project_dataset, numeric_feature_columns
from train import build_numeric_only_preprocessor


def build_base_models(random_state: int = 42) -> list[tuple[str, object]]:
    return [
        (
            "random_forest",
            RandomForestClassifier(n_estimators=80, min_samples_leaf=3, class_weight="balanced", random_state=random_state, n_jobs=-1),
        ),
        (
            "svm",
            SVC(kernel="rbf", C=2.0, gamma="scale", probability=True, class_weight="balanced", random_state=random_state),
        ),
        (
            "xgboost",
            XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9, eval_metric="logloss", random_state=random_state),
        ),
        (
            "mlp",
            MLPClassifier(hidden_layer_sizes=(32,), alpha=0.001, early_stopping=True, validation_fraction=0.15, max_iter=300, random_state=random_state),
        ),
    ]


def build_stacking_pipeline(columns: list[str], final_estimator: object, random_state: int = 42) -> Pipeline:
    stacking = StackingClassifier(
        estimators=build_base_models(random_state),
        final_estimator=final_estimator,
        stack_method="predict_proba",
        cv=3,
        n_jobs=-1,
    )
    return Pipeline(
        steps=[
            ("preprocessor", build_numeric_only_preprocessor(columns)),
            ("model", stacking),
        ]
    )


def evaluate_model(name: str, model: Pipeline, X_train, X_test, y_train, y_test) -> dict[str, float | str]:
    fitted = clone(model)
    fitted.fit(X_train, y_train)
    pred = fitted.predict(X_test)
    return {
        "model": name,
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, pred, zero_division=0)), 4),
    }, fitted


def lime_or_local_explanation(model: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    sample = X_test.iloc[0]
    try:
        from lime.lime_tabular import LimeTabularExplainer

        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["model"]
        train_ready = preprocessor.transform(X_train)
        sample_ready = preprocessor.transform(pd.DataFrame([sample]))
        explainer = LimeTabularExplainer(
            training_data=np.asarray(train_ready),
            feature_names=columns,
            class_names=["safe", "unsafe"],
            mode="classification",
            discretize_continuous=True,
        )
        explanation = explainer.explain_instance(
            np.asarray(sample_ready)[0],
            classifier.predict_proba,
            num_features=12,
        )
        return pd.DataFrame(explanation.as_list(), columns=["feature_condition", "lime_weight"])
    except Exception:
        result = permutation_importance(model, X_test.head(80), model.predict(X_test.head(80)), n_repeats=5, random_state=42)
        order = np.argsort(result.importances_mean)[::-1][:12]
        return pd.DataFrame(
            {
                "feature_condition": [columns[index] for index in order if index < len(columns)],
                "lime_weight": [float(result.importances_mean[index]) for index in order if index < len(columns)],
            }
        )


def main() -> None:
    output_dir = ensure_output_dir("lab09")
    df = load_project_dataset()
    columns = numeric_feature_columns(df)
    X = df.drop(columns=["label"])
    y = df["label"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    final_estimators = {
        "stacking_logistic_meta": LogisticRegression(max_iter=3000, random_state=42),
        "stacking_random_forest_meta": RandomForestClassifier(n_estimators=100, min_samples_leaf=3, random_state=42),
    }

    rows: list[dict[str, float | str]] = []
    fitted_models: dict[str, Pipeline] = {}
    for name, final_estimator in final_estimators.items():
        row, fitted = evaluate_model(name, build_stacking_pipeline(columns, final_estimator), X_train, X_test, y_train, y_test)
        rows.append(row)
        fitted_models[name] = fitted

    soft_voting = Pipeline(
        steps=[
            ("preprocessor", build_numeric_only_preprocessor(columns)),
            (
                "model",
                VotingClassifier(
                    estimators=build_base_models(),
                    voting="soft",
                    weights=[2, 2, 2, 1],
                    n_jobs=-1,
                ),
            ),
        ]
    )
    row, fitted_voting = evaluate_model("weighted_soft_voting", soft_voting, X_train, X_test, y_train, y_test)
    rows.append(row)
    fitted_models["weighted_soft_voting"] = fitted_voting

    results = pd.DataFrame(rows).sort_values(["accuracy", "f1_score"], ascending=False)
    results.to_csv(output_dir / "stacking_pipeline_results.csv", index=False)

    best_name = str(results.iloc[0]["model"])
    explanation = lime_or_local_explanation(fitted_models[best_name], X_train[columns], X_test[columns], columns)
    explanation.to_csv(output_dir / "lime_explanation_best_pipeline.csv", index=False)

    summary = {
        "best_model": best_name,
        "best_accuracy": float(results.iloc[0]["accuracy"]),
        "pipeline_steps": ["missing value imputation", "standard scaling", "stacking/voting classifier"],
        "stacking_base_models": ["Random Forest", "SVM", "XGBoost", "MLP Classifier"],
        "meta_learners_tested": list(final_estimators),
        "lime_note": "Uses LIME when installed; otherwise saves local permutation-style explanation as fallback.",
    }
    (output_dir / "lab09_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Lab09 completed.")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
