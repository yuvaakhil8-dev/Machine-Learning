# Model Documentation

This repository compares multiple supervised classifiers for Safe/Unsafe video classification using extracted video-level features.

## Logistic Regression

Source: `Source_Code/models/logistic_regression.py`

Logistic Regression is a linear classifier used as a baseline. It models the probability of a class using a linear decision boundary over standardized feature values.

## Decision Tree

Source: `Source_Code/models/decision_tree.py`

Decision Tree learns rule-based splits over feature values. It is interpretable but can overfit high-dimensional feature spaces if not regularized.

## Random Forest

Source: `Source_Code/models/random_forest.py`

Random Forest combines multiple decision trees trained on bootstrapped samples and random feature subsets. It is useful for nonlinear feature interactions and feature-importance analysis.

## Support Vector Machine

Source: `Source_Code/models/svm_model.py`

SVM is effective for high-dimensional feature spaces. In the available results, SVM is the best test-set model with 0.8467 accuracy and 0.9278 ROC-AUC.

## K-Nearest Neighbors

Source: `Source_Code/models/knn_model.py`

KNN predicts by comparing a sample to nearby examples in feature space. It benefits from standardized features and is included as a distance-based baseline.

## Naive Bayes

Source: `Source_Code/models/naive_bayes.py`

Naive Bayes is a probabilistic baseline that assumes conditional independence between features. It is simple and fast, but the independence assumption may limit performance on correlated visual features.

## AdaBoost

Source: `Source_Code/models/adaboost.py`

AdaBoost trains a sequence of weak learners and increases focus on previously misclassified examples. It is included as a boosting baseline.

## XGBoost

Source: `Source_Code/models/xgboost_model.py`

XGBoost is a gradient-boosted tree model. It is included for strong nonlinear classification and has optimized artifacts in `Models/optimization/`.

## MLP Classifier

Source: `Source_Code/models/mlp_classifier.py`

MLP Classifier is a feed-forward neural network trained on the extracted feature vectors. It provides a nonlinear neural baseline over the engineered feature representation.

## Stacking Classifier

Source: `Source_Code/models/stacking_classifier.py`

Stacking combines multiple model outputs through a meta-model. In the available results, the stacking classifier reaches 0.8333 test accuracy and 0.9148 ROC-AUC.

## Model Artifacts

Saved model files are stored in:

- `Models/`
- `Models/artifacts_phase1/`
- `Models/optimization/`
