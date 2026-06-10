# Results Summary

This file summarizes the available evaluation outputs in the repository. The values are taken from `Results/metrics_summary.csv`, `Results/classifier_comparison.csv`, `Results/cross_validation_results.csv`, and `Results/generated_project_outputs/model_ranking.csv`.

## Best Model

The best test-set model in the available result files is:

| Metric | Value |
|---|---:|
| Model | SVM |
| Test accuracy | 0.8467 |
| Precision | 0.8514 |
| Recall | 0.8400 |
| F1-score | 0.8456 |
| ROC-AUC | 0.9278 |
| PR-AUC | 0.9206 |

## Classifier Comparison

| Model | Test Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| SVM | 0.8467 | 0.8514 | 0.8400 | 0.8456 | 0.9278 |
| Stacking Classifier | 0.8333 | 0.8571 | 0.8000 | 0.8276 | 0.9148 |
| KNN | 0.8200 | 0.7857 | 0.8800 | 0.8302 | 0.8835 |
| MLP Classifier | 0.8133 | 0.7831 | 0.8667 | 0.8228 | 0.8889 |
| XGBoost | 0.8067 | 0.8108 | 0.8000 | 0.8054 | 0.8948 |
| Logistic Regression | 0.8000 | 0.7848 | 0.8267 | 0.8052 | 0.8580 |
| Random Forest | 0.8000 | 0.8169 | 0.7733 | 0.7945 | 0.8866 |
| Naive Bayes | 0.7800 | 0.7838 | 0.7733 | 0.7785 | 0.8408 |
| AdaBoost | 0.7800 | 0.8000 | 0.7467 | 0.7724 | 0.8631 |
| Decision Tree | 0.7733 | 0.8254 | 0.6933 | 0.7536 | 0.8012 |

## Cross-Validation Summary

The project includes 5-fold cross-validation results for all major models.

| Model | CV Accuracy Mean | CV Accuracy Std | CV F1 Mean | CV ROC-AUC Mean |
|---|---:|---:|---:|---:|
| SVM | 0.8833 | 0.0119 | 0.8844 | 0.9463 |
| XGBoost | 0.8653 | 0.0310 | 0.8693 | 0.9229 |
| Stacking Classifier | 0.8492 | 0.0431 | 0.8583 | 0.9291 |
| Random Forest | 0.8431 | 0.0264 | 0.8430 | 0.9311 |
| KNN | 0.8331 | 0.0353 | 0.8431 | 0.9013 |

## ROC-AUC Summary

The highest ROC-AUC is reported by SVM at 0.9278 on the test set and 0.9463 mean ROC-AUC under 5-fold cross-validation.

## Confusion Matrix Summary

SVM classification report from `Models/artifacts_phase1/phase4_visual_svm_classification_report.txt`:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| safe | 0.84 | 0.85 | 0.85 | 75 |
| unsafe | 0.85 | 0.84 | 0.85 | 75 |

The confusion matrix image is available at `screenshots/confusion_matrix_svm.png`.

## Major Observations

- SVM is the strongest model in the available test and cross-validation outputs.
- The dataset is balanced, with 250 safe videos and 247 unsafe videos.
- Stacking ensemble learning is implemented and performs second by test accuracy.
- KNN shows the highest recall among the listed top models at 0.8800.
- Explainability artifacts are available through LIME and SHAP outputs.
