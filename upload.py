"""Upload a complete runtime package to the existing Hugging Face Space."""

from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

REPO_ID = "muqing-research/nsclc-survival"
FILES = (
    "app.py",
    "survival_app.py",
    "survival_core.py",
    "tcga_luad_app_bundle.pkl",
    "eda_decisions.json",
    "theme.css",
    "world.geojson",
    "requirements.txt",
    "Dockerfile",
)


def main() -> None:
    folder = Path(__file__).resolve().parent
    missing = [name for name in FILES if not (folder / name).is_file()]
    if missing:
        raise RuntimeError("Missing deployment files: " + ", ".join(missing))
    message = sys.argv[1] if len(sys.argv) > 1 else "Update application and model bundle"
    api = HfApi()
    api.repo_info(repo_id=REPO_ID, repo_type="space")
    # One commit keeps application code, stylesheet, and model versions together.
    api.create_commit(
        repo_id=REPO_ID,
        repo_type="space",
        operations=[
            CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(folder / name))
            for name in FILES
        ],
        commit_message=message,
    )
    print(f"Uploaded {len(FILES)} files: https://huggingface.co/spaces/{REPO_ID}")


if __name__ == "__main__":
    main()
