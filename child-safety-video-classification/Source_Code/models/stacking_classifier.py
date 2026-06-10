"""Stacking classifier model factory.

Purpose:
    Expose the stacking classifier builder used by the training script without
    duplicating the ensemble configuration.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "Source_Code" / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from train import build_stacking_classifier


def build_model(random_state: int = 42):
    """Return the configured stacking classifier."""
    return build_stacking_classifier(random_state)
