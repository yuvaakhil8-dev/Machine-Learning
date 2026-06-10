from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


EXCLUDED_COLUMNS = {"label", "video_name", "video_path", "class_name"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight hyperparameter optimization for key classifiers.")
    parser.add_argument("--csv", default="Feature_Files/video_features_advanced.csv")
    parser.add_argument("--output-dir", default="Models/optimization")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--folds", type=int, default=3)
    return parser.parse_args()


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in df.columns
        if column not in EXCLUDED_COLUMNS and pd.api.types.is_numeric_dtype(df[column])
    ]


def build_preprocessor(columns: list[str]) -> ColumnTransformer:
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


def optimize_models(df: pd.DataFrame, output_dir: Path, random_state: int, folds: int) -> pd.DataFrame:
    columns = numeric_columns(df)
    X = df.drop(columns=["label"])
    y = df["label"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, stratify=y, random_state=random_state)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)

    preprocessor = build_preprocessor(columns)
    searches = {
        "grid_logistic_regression": GridSearchCV(
            Pipeline([("preprocessor", preprocessor), ("model", LogisticRegression(max_iter=3000, random_state=random_state))]),
            param_grid={"model__C": [0.1, 1.0, 3.0], "model__class_weight": [None, "balanced"]},
            scoring="f1",
            cv=cv,
            n_jobs=1,
        ),
        "grid_svm": GridSearchCV(
            Pipeline([("preprocessor", preprocessor), ("model", SVC(kernel="rbf", probability=True, random_state=random_state))]),
            param_grid={"model__C": [0.5, 1.0, 2.0], "model__class_weight": [None, "balanced"]},
            scoring="f1",
            cv=cv,
            n_jobs=1,
        ),
        "randomized_random_forest": RandomizedSearchCV(
            Pipeline([("preprocessor", preprocessor), ("model", RandomForestClassifier(class_weight="balanced", random_state=random_state, n_jobs=-1))]),
            param_distributions={
                "model__n_estimators": [150, 220, 300],
                "model__max_depth": [None, 8, 12],
                "model__min_samples_leaf": [1, 2, 3],
            },
            n_iter=6,
            scoring="f1",
            cv=cv,
            random_state=random_state,
            n_jobs=1,
        ),
        "randomized_xgboost": RandomizedSearchCV(
            Pipeline(
                [
                    ("preprocessor", preprocessor),
                    (
                        "model",
                        XGBClassifier(
                            eval_metric="logloss",
                            random_state=random_state,
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
            param_distributions={
                "model__n_estimators": [120, 180, 250],
                "model__max_depth": [3, 4, 5],
                "model__learning_rate": [0.03, 0.05, 0.08],
                "model__subsample": [0.8, 0.9, 1.0],
            },
            n_iter=6,
            scoring="f1",
            cv=cv,
            random_state=random_state,
            n_jobs=1,
        ),
    }

    rows = []
    for name, search in searches.items():
        search.fit(X_train, y_train)
        joblib.dump(search.best_estimator_, output_dir / f"{name}_best_model.joblib")
        rows.append(
            {
                "search_name": name,
                "best_cv_f1": round(float(search.best_score_), 4),
                "best_params": search.best_params_,
            }
        )

    results = pd.DataFrame(rows).sort_values("best_cv_f1", ascending=False)
    results.to_csv(output_dir / "hyperparameter_optimization_results.csv", index=False)
    return results


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    results = optimize_models(df, output_dir, args.random_state, args.folds)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
