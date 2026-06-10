# Evaluation Metrics

This document describes the evaluation metrics used by the project.

## Accuracy

Accuracy measures the fraction of total predictions that are correct.

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

## Precision

Precision measures how many predicted positive samples are actually positive.

```text
Precision = TP / (TP + FP)
```

## Recall

Recall measures how many actual positive samples are correctly detected.

```text
Recall = TP / (TP + FN)
```

## F1 Score

F1-score is the harmonic mean of precision and recall.

```text
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

## Confusion Matrix

A confusion matrix summarizes correct and incorrect predictions by class.

| Term | Meaning |
|---|---|
| TP | Positive class predicted as positive |
| TN | Negative class predicted as negative |
| FP | Negative class predicted as positive |
| FN | Positive class predicted as negative |

Confusion matrix images are stored in:

`Results/confusion_matrices/`

## ROC Curve

The ROC curve plots true positive rate against false positive rate at different decision thresholds.

```text
True Positive Rate = TP / (TP + FN)
False Positive Rate = FP / (FP + TN)
```

## ROC-AUC

ROC-AUC is the area under the ROC curve. A higher value indicates stronger ranking ability between classes.

The best available ROC-AUC result is SVM with 0.9278.

## Precision-Recall Curve

The Precision-Recall curve shows the tradeoff between precision and recall across thresholds. It is useful when class-wise detection quality matters.

## Cross Validation

Cross-validation evaluates model generalization across multiple train/validation splits.

This project includes 5-fold cross-validation results in:

`Results/cross_validation_results.csv`

For k-fold cross-validation:

```text
CV Score = mean(score_1, score_2, ..., score_k)
CV Std = standard deviation of fold scores
```
