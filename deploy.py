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
    "app.py",
    "chart_views.py",
    "survival_app.py",
    "survival_core.py",
    "tcga_luad_app_bundle.pkl",
    "eda_decisions.json",
    "theme.css",
    "world.geojson",
    "requirements.txt",
)


def validate_package() -> list[str]:
    """Require a complete package and the tested Python/dependency versions."""
    missing = [name for name in RUNTIME_FILES if not (ROOT / name).is_file()]
    if missing:
        raise RuntimeError("Missing runtime files: " + ", ".join(missing))
    if sys.version_info[:2] != (3, 13):
        raise RuntimeError("Deploy with the tested Python 3.13 environment.")
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        requirement = line.strip()
        if not requirement or requirement.startswith("#"):
            continue
        name, expected = requirement.split("==", 1)
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise RuntimeError(f"{name}: expected {expected}, found {actual}")
    return sorted(path.name for path in ROOT.iterdir() if path.name not in RUNTIME_FILES)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate without uploading.")
    args = parser.parse_args()
    excludes = validate_package()
    print("Runtime files: " + ", ".join(RUNTIME_FILES), flush=True)
    if args.check:
        print("Deployment preflight passed.")
        return 0
    cmd = [
        sys.executable, "-m", "rsconnect.main", "deploy", "shiny", str(ROOT),
        "--name", ACCOUNT, "--title", TITLE, "--app-id", APP_ID,
        "--python", sys.executable,
    ]
    for name in excludes:
        cmd.extend(("--exclude", name))
    # shinyapps.io does not support rsconnect --environment management.
    print(f"Deploying https://{ACCOUNT}.shinyapps.io/{TITLE}/", flush=True)
    return subprocess.run(cmd, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
