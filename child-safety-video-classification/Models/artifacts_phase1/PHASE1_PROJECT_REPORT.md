# Project Report: Safe and Unsafe Video Activity Classification

## Dataset and Preprocessing
- Dataset file: `data\features\video_features_advanced.csv`
- Total videos represented: 497
- Safe videos (`label=0`): 250
- Unsafe videos (`label=1`): 247
- Video/CNN features used: 515
- Missing values are handled with `SimpleImputer`.
- Numeric video features are standardized with `StandardScaler`.
- The train/test split is stratified, so both classes remain balanced in evaluation.

## Feature Reliability
This project does **not** decide child safety using only brightness, darkness, or video color.
The main visual representation is a pretrained CNN embedding with named groups such as `edge_texture_feature_001`, `object_body_feature_001`, `scene_activity_feature_001`, and `safety_context_feature_001`.
Brightness alone is not used as a safety rule because a black shirt, black paint, or a dark background does not make a video unsafe.

## Models Implemented
- Logistic Regression baseline
- Decision Tree
- Random Forest
- Support Vector Machine with RBF kernel
- XGBoost
- K-Nearest Neighbors
- Naive Bayes
- AdaBoost
- MLP Classifier
- KNN sweep for comparison
- Stacking Classifier combining Random Forest, SVM, XGBoost, and MLP with Logistic Regression as the final estimator

## Best Result
- Best model: `phase4_visual_svm`
- Test accuracy: 0.8467
- Precision: 0.8514
- Recall: 0.8400
- F1 score: 0.8456
- Generalization status: `cv_review`

## Analysis
- Inter-class distance: 14.3359
- Mean intra-class spread: 20.2849
- Separability verdict: weak

## Conclusion
The project demonstrates dataset preparation, preprocessing, multiple ML models, the required stacking classifier, and result analysis with saved metrics, reports, predictions, model files, and confusion matrices.
