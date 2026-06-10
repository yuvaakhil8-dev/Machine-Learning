"""CNN feature extraction adapter.

Purpose:
    Import the visual feature extractor from the main multi-model pipeline for
    reuse by modules that need CNN embeddings.

Inputs:
    Sampled and preprocessed video frames.

Outputs:
    CNN embedding features through `extract_visual_features`.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "Source_Code" / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from multi_model_pipeline import extract_visual_features, load_visual_model
