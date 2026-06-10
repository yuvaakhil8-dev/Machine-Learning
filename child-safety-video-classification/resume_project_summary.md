# Project Summary

## 2-Line Version

Built a computer vision pipeline for Safe/Unsafe video classification using preprocessing, CNN embeddings, YOLO/object cues, motion/activity features, feature fusion, and supervised machine learning.

Compared 10 classifiers with train/test evaluation and 5-fold cross-validation; the best available test-set model is SVM with 0.8467 accuracy, 0.8456 F1-score, and 0.9278 ROC-AUC.

## 3-Bullet Version

- Developed an end-to-end video classification repository with preprocessing, frame extraction, feature extraction, feature fusion, model training, evaluation, explainability, and Streamlit dashboard components.
- Compared Logistic Regression, Decision Tree, Random Forest, SVM, KNN, Naive Bayes, AdaBoost, XGBoost, MLP, and Stacking Classifier on 497 processed video samples.
- Reported model performance with accuracy, precision, recall, F1-score, ROC-AUC, PR-AUC, confusion matrices, 5-fold cross-validation, LIME, and SHAP outputs.

## 5-Bullet Version

- Implemented a Safe/Unsafe video classification pipeline using OpenCV-based preprocessing, frame sampling, resizing, normalization, and video metadata validation.
- Extracted video-level feature representations using CNN embeddings, motion descriptors, YOLO/object features, expression proxies, pose/activity proxies, and feature fusion.
- Trained and evaluated 10 supervised classifiers, including SVM, Random Forest, XGBoost, MLP, and Stacking Ensemble models.
- Achieved best available test-set performance with SVM: 0.8467 accuracy, 0.8456 F1-score, and 0.9278 ROC-AUC.
- Included evaluation and explainability artifacts such as confusion matrices, ROC curves, Precision-Recall curves, PCA plots, permutation importance, LIME explanations, SHAP summaries, and a Streamlit dashboard.

## Technical Keywords

Computer Vision, Video Classification, Safe/Unsafe Classification, OpenCV, CNN Feature Extraction, YOLOv8, Motion Features, Feature Fusion, SVM, Random Forest, XGBoost, MLP, Stacking Ensemble, Cross-Validation, ROC-AUC, Precision-Recall, Confusion Matrix, LIME, SHAP, Streamlit, scikit-learn, PyTorch.
