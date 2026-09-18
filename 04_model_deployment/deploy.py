# Module guide:
# - Role: Validate and publish the survival Shiny application.
# - Workflow: Check required runtime files before delegating to the deployment command.
# - Design note: Fail early when package contents or runtime compatibility are invalid.
"""Validate the runtime package and deploy the existing shinyapps.io app."""

from __future__ import annotations

import argparse
import importlib.metadata
import subprocess
import sys
from pathlib import Path

ACCOUNT = "medictio"
TITLE = "nsclc-survival"
APP_ID = "17579074"
ROOT = Path(__file__).resolve().parent
RUNTIME_FILES = (
    ".python-version",
    "app.py",
    "visualizations.py",
    "clean_clinical_data.py",
    "survival_preprocessing.py",
    "survival_app.py",
    "survival_core.py",
    "tcga_luad_survival_model_bundle.pkl",
    "preprocessing_decisions.json",
    "compact_theme.css",
    "world.geojson",
    "requirements.txt",
)
LOCAL_UPLOAD_PYTHON_VERSIONS = {(3, 13), (3, 12)}


# Function guide: validate_package is responsible for validate package.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def validate_package() -> None:
    """Require a complete package and the tested Python/dependency versions."""
    missing = [name for name in RUNTIME_FILES if not (ROOT / name).is_file()]
    if missing:
        raise RuntimeError("Missing runtime files: " + ", ".join(missing))
    if sys.version_info[:2] not in LOCAL_UPLOAD_PYTHON_VERSIONS:
        raise RuntimeError("Deploy with the tested Python 3.12 or 3.13 environment.")
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        requirement = line.strip()
        if not requirement or requirement.startswith("#"):
            continue
        name, expected = requirement.split("==", 1)
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise RuntimeError(f"{name}: expected {expected}, found {actual}")


# Function guide: main is responsible for run the module main workflow.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate without uploading.")
    args = parser.parse_args()
    validate_package()
    print("Runtime files: " + ", ".join(RUNTIME_FILES), flush=True)
    if args.check:
        print("Deployment preflight passed.")
        return 0
    cmd = [
        sys.executable, "-m", "rsconnect.main", "deploy", "shiny", str(ROOT),
        *RUNTIME_FILES,
        "--name", ACCOUNT, "--title", TITLE, "--app-id", APP_ID,
        "--python", sys.executable,
        "--exclude", "**",
    ]
    # shinyapps.io does not support rsconnect --environment management.
    print(f"Deploying https://{ACCOUNT}.shinyapps.io/{TITLE}/", flush=True)
    return subprocess.run(cmd, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
