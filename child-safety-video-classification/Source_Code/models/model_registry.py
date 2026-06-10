"""Classical model registry.

Purpose:
    Centralize model construction so training scripts can request a consistent
    set of baseline and ensemble classifiers.

Inputs:
    A random seed for reproducible model initialization.

Outputs:
    A dictionary mapping model names to initialized estimator instances.
"""

from __future__ import annotations

from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None


def classical_model_registry(random_state: int = 42) -> dict[str, object]:
    """Return the configured classical ML models used by the project."""
    models = {
        "logistic_regression": LogisticRegression(max_iter=3000, random_state=random_state),
        "naive_bayes": GaussianNB(),
        "decision_tree": DecisionTreeClassifier(random_state=random_state, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "knn": KNeighborsClassifier(n_neighbors=9),
        "svm": SVC(kernel="rbf", C=2.0, probability=True, class_weight="balanced", random_state=random_state),
        "adaboost": AdaBoostClassifier(n_estimators=150, random_state=random_state),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
        "xgboost": XGBClassifier(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
        ),
    }
    if LGBMClassifier is not None:
        models["lightgbm"] = LGBMClassifier(
            n_estimators=250,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=random_state,
            verbose=-1,
        )
    return models
