"""Check canonical artifacts without overwriting the hosted app sources."""

# Module guide:
# - Role: Preserve a legacy artifact-synchronization check.
# - Workflow: Inspect canonical artifacts without overwriting hosted application sources.
# - Design note: Keep file operations narrow and auditable.
from pathlib import Path

if __name__ == "__main__":
    canonical = Path(__file__).resolve().parents[2]
    deployment = canonical / "04_model_deployment"
    required = (
        "survival_app.py", "survival_core.py",
        "tcga_luad_survival_model_bundle.pkl", "preprocessing_decisions.json",
    )
    missing = [name for name in required if not (deployment / name).is_file()]
    if missing:
        raise SystemExit(
            f"Missing canonical artifacts: {missing}. Run "
            "03_model_training_and_evaluation/train_and_export_model_bundle.py."
        )
    print(f"Canonical sources are ready at {deployment}; no copy is needed.")
