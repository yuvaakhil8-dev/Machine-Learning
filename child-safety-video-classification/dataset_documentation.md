# Dataset Documentation

## Dataset Structure

The processed project dataset is represented as a feature CSV:

`Feature_Files/video_features_advanced_descriptive_with_dimensions.csv`

The file contains one row per video and includes metadata columns, class labels, preprocessing proof columns, and extracted feature columns.

## Class Labels

| Class | Label |
|---|---:|
| safe | 0 |
| unsafe | 1 |

## Class Distribution

| Class | Count |
|---|---:|
| safe | 250 |
| unsafe | 247 |
| Total | 497 |

## Dataset Statistics

| Item | Value |
|---|---:|
| Rows/videos | 497 |
| Current main feature CSV columns | 533 |
| Numeric feature columns reported by project audit | 515 |
| Safe videos | 250 |
| Unsafe videos | 247 |

## Dataset Audit Files

- `Results/video_quality_audit.csv`
- `Results/video_quality_summary.json`
- `Feature_Files/01_Same_Dimensions_And_Normalization_Proof.csv`
- `Feature_Files/02_Feature_Dataset_Shape_Summary.csv`

## Notes

The processed feature file is the dataset used by the available model training and evaluation artifacts. Raw videos are not required for reading the included evaluation outputs, but raw video files would be required to rerun the complete feature-extraction process.
