# Module guide:
# - Role: Preserve a legacy survival workflow.
# - Workflow: Retain historical deployment or synchronization behavior for comparison.
# - Design note: Review archived assumptions before using a routine in a new pipeline.
"""Deploy the canonical, validated survival application package."""

import runpy
from pathlib import Path

if __name__ == "__main__":
    canonical = Path(__file__).resolve().parents[2]
    runpy.run_path(
        str(canonical / "04_model_deployment" / "deploy.py"),
        run_name="__main__",
    )
