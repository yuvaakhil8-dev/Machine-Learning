# Repository Structure

This document explains the current repository folders and major files.

## Root Files

| File | Purpose |
|---|---|
| `README.md` | Main GitHub-facing project documentation |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Files and folders excluded from Git tracking |
| `architecture_diagram.png` | Generated architecture diagram |
| `results_summary.md` | Summary of available model results |
| `model_documentation.md` | Model descriptions |
| `feature_documentation.md` | Feature extraction and fusion documentation |
| `evaluation_metrics.md` | Evaluation metric definitions |
| `dataset_documentation.md` | Dataset structure and statistics |
| `repository_structure.md` | Repository organization guide |
| `resume_project_summary.md` | Project summary in concise formats |

## Feature_Files

Contains processed feature datasets and preprocessing proof files.

Important files:

- `video_features_advanced_descriptive_with_dimensions.csv`
- `01_Same_Dimensions_And_Normalization_Proof.csv`
- `02_Feature_Dataset_Shape_Summary.csv`

## Models

Contains trained model files, optimization outputs, and phase-wise artifacts.

Important files:

- `scaler.joblib`
- `random_forest.joblib`
- `svm_model.joblib`
- `xgboost_model.joblib`
- `mlp_model.joblib`
- `stacking_model.joblib`
- `yolov8n.pt`

Subfolders:

- `Models/artifacts_phase1/`: phase-wise trained models, classification reports, predictions, and confusion matrices
- `Models/optimization/`: hyperparameter-optimized model artifacts and optimization results

## Presentation

Contains PowerPoint and presentation-ready visual assets.

Important files:

- `Final_Project_Presentation.pptx`
- `architecture_diagram.png`
- `classifier_comparison.png`
- `confusion_matrix.png`
- `feature_importance.png`
- `pca_visualization.png`
- `roc_curve.png`

## Results

Contains evaluation outputs, model comparison tables, charts, and generated analysis artifacts.

Important files:

- `metrics_summary.csv`
- `classifier_comparison.csv`
- `cross_validation_results.csv`
- `video_quality_audit.csv`
- `video_quality_summary.json`

Subfolders:

- `confusion_matrices/`
- `feature_importance/`
- `generated_project_outputs/`
- `pca_plots/`
- `roc_curves/`
- `screenshots/`

## Source_Code

Contains all Python source code.

Subfolders:

- `preprocessing/`: video validation, resizing, normalization, augmentation, metadata, and dataset preparation
- `feature_engineering/`: CNN, motion, YOLO, expression, pose/activity, fusion, and standardization features
- `models/`: model definitions and optimization
- `evaluation/`: metrics, confusion matrices, ROC-AUC, PCA, cross-validation, feature importance, and error analysis
- `scripts/`: runnable training, evaluation, audit, and report-generation scripts
- `streamlit_app/`: Streamlit dashboard code
- `utils/`: shared helpers, config, feature names, file utilities, and video I/O

## screenshots

Contains selected visual assets for README and documentation:

- `confusion_matrix_svm.png`
- `roc_curve.png`
- `precision_recall_curve.png`
- `classifier_comparison.png`
- `pca_visualization.png`
- `feature_importance.png`
- `shap_kernel_importance_top20.png`
- `streamlit_report_screenshot.png`
