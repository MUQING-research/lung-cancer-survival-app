# Module guide:
# - Role: Train, evaluate, or test the survival modelling workflow.
# - Workflow: Compare Cox, AFT, and related survival utilities on train and test partitions and validate visualizations.
# - Design note: Keep regression tests focused on reproducible bundle contents and evaluation contracts.
"""Train the survival models and export the deployment model bundle."""

import os
import sys
from pathlib import Path


DEPLOYMENT_DIR = Path(__file__).resolve().parents[1] / "04_model_deployment"
TRAINING_CACHE_DIR = Path(__file__).resolve().parent / ".cache"
PREPROCESSING_DIR = Path(__file__).resolve().parents[1] / "02_data_preprocessing"
DATA_CLEANING_DIR = Path(__file__).resolve().parents[1] / "01_data_cleaning"
sys.path.insert(0, str(DEPLOYMENT_DIR))
sys.path.insert(0, str(PREPROCESSING_DIR))
sys.path.insert(0, str(DATA_CLEANING_DIR))


# Function guide: main is responsible for run the module main workflow.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def main():
    os.environ["SURVIVAL_REBUILD_BUNDLE"] = "1"
    os.environ["SURVIVAL_TRAINING_CACHE_DIR"] = str(TRAINING_CACHE_DIR)
    os.environ["SURVIVAL_CACHE_DIR"] = str(TRAINING_CACHE_DIR)
    os.environ["SURVIVAL_AUDIT_DIR"] = str(TRAINING_CACHE_DIR / "audit")
    from clean_clinical_data import (
        clean_tcga_luad_records,
        download_tcga_luad_records,
        write_cleaning_outputs,
    )

    source_cache = TRAINING_CACHE_DIR / "tcga_luad_gdc.json"
    cleaned = clean_tcga_luad_records(
        download_tcga_luad_records(source_cache)
    )
    write_cleaning_outputs(cleaned, TRAINING_CACHE_DIR)

    import survival_app

    bundle = survival_app._B
    for model in ("cox", "aft"):
        train, test = bundle[f"tr_{model}"], bundle[f"res_{model}"]
        print(f"{model}: train/test C-index={train['c_index']:.6f}/{test['c_index']:.6f}; "
              f"train/test IBS={train['ibs']:.6f}/{test['ibs']:.6f}")
    print(f"Saved model bundle: {survival_app._BUNDLE_PATH}")


if __name__ == "__main__":
    main()
