"""Compatibility entrypoint for the canonical hosted survival application."""

# Module guide:
# - Role: Preserve a legacy survival application entry point.
# - Workflow: Import and expose the canonical hosted survival application.
# - Design note: Keep this compatibility wrapper separate from the active app module.
import runpy
import sys
from pathlib import Path

canonical = Path(__file__).resolve().parents[2]
deployment = canonical / "04_model_deployment"
sys.path.insert(0, str(deployment))
app = runpy.run_path(str(deployment / "survival_app.py"))["app"]
