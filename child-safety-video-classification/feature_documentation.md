# Feature Documentation

The project converts video files into structured feature vectors for classical machine learning and ensemble classification.

## CNN Features

CNN feature extraction is implemented in:

- `Source_Code/feature_engineering/cnn_feature_extraction.py`
- `Source_Code/feature_engineering/cnn_feature_names.py`
- `Source_Code/models/transfer_backbones.py`

The feature dictionary is available at:

`Results/generated_project_outputs/feature_dictionary_used_by_project.csv`

The repository uses descriptive names for learned CNN embedding dimensions. These names make reports easier to read, but they are not exact manual detectors.

## Motion Features

Motion features are implemented in:

`Source_Code/feature_engineering/video_features.py`

These features describe temporal movement and activity changes across sampled video frames.

## YOLO Features

YOLO-related feature code is implemented in:

- `Source_Code/feature_engineering/yolo_object_features.py`

The YOLO model file is stored at:

`Models/yolov8n.pt`

These features support object/person-related visual cues.

## Expression Features

Expression-related features are implemented in:

- `Source_Code/feature_engineering/emotion_features.py`

These modules provide expression and emotion proxy descriptors extracted from visual patterns.

## Activity Features

Activity and pose-related features are implemented in:

`Source_Code/feature_engineering/pose_activity_features.py`

These features represent activity, posture, and movement proxy information.

## Safety-Context Features

Safety-context feature names appear in:

`Results/generated_project_outputs/feature_dictionary_used_by_project.csv`

They are used as readable descriptions for learned or fused video-level feature signals associated with risk, safety, interaction, motion, and scene context.

## Feature Fusion

Feature fusion is implemented in:

`Source_Code/feature_engineering/video_fusion.py`

The fusion stage combines multiple signal families into final video-level features used by the classifiers.

## Feature Dataset

Main processed feature file:

`Feature_Files/video_features_advanced_descriptive_with_dimensions.csv`

This file contains 497 rows and 533 columns in the current repository copy.
