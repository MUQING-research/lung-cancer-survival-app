"""Build the canonical hosted survival app bundle."""

# Module guide:
# - Role: Preserve a legacy survival bundle-building entry point.
# - Workflow: Delegate bundle construction to the canonical training script.
# - Design note: Keep this compatibility wrapper separate from the active deployment path.
import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    canonical = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(canonical))
    runpy.run_path(
        str(canonical / "03_model_training_and_evaluation" / "train_and_export_model_bundle.py"),
        run_name="__main__",
    )
