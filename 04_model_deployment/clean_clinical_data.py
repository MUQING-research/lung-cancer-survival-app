# Module guide:
# - Role: Define the deployed survival Shiny application.
# - Workflow: Load a precomputed bundle, validate clinical inputs, estimate survival, and render diagnostics.
# - Design note: The deployed process must not depend on raw training feature matrices.
"""Download and clean TCGA-LUAD clinical records before model preprocessing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests


TIME_COLUMN = "time_months"
EVENT_COLUMN = "event_death"
GDC_URL = "https://api.gdc.cancer.gov/cases"
DAYS_PER_MONTH = 365.25 / 12.0

GDC_FIELDS = ",".join([
    "case_id", "demographic.vital_status", "demographic.days_to_death",
    "diagnoses.days_to_last_follow_up", "diagnoses.days_to_death",
    "diagnoses.age_at_diagnosis", "diagnoses.ajcc_pathologic_stage",
    "diagnoses.ajcc_pathologic_t", "diagnoses.ajcc_pathologic_n",
    "diagnoses.ajcc_pathologic_m",
])

STAGE_MAP = {
    "Stage I": 1, "Stage IA": 1, "Stage IA1": 1, "Stage IA2": 1,
    "Stage IA3": 1, "Stage IB": 1,
    "Stage II": 2, "Stage IIA": 2, "Stage IIB": 2,
    "Stage III": 3, "Stage IIIA": 3, "Stage IIIB": 3, "Stage IIIC": 3,
    "Stage IV": 4, "Stage IVA": 4, "Stage IVB": 4,
}
T_STAGE_MAP = {
    "T0": 0, "T1": 1, "T1a": 1, "T1b": 1, "T1c": 1, "T1mi": 1,
    "T2": 2, "T2a": 2, "T2b": 2, "T3": 3, "T4": 4,
}
N_STAGE_MAP = {"N0": 0, "N1": 1, "N2": 2, "N3": 3}

DATA_DICTIONARY = [
    {"name": "case_id", "role": "identifier", "description": "TCGA case identifier"},
    {"name": TIME_COLUMN, "role": "outcome_time", "unit": "months", "description": "Observed overall survival time"},
    {"name": EVENT_COLUMN, "role": "event_indicator", "coding": "1=death, 0=right-censored", "description": "Observed death indicator"},
    {"name": "age", "role": "predictor", "type": "continuous", "unit": "years", "description": "Age at diagnosis"},
    {"name": "stage", "role": "predictor", "type": "ordinal", "coding": "1=I, 2=II, 3=III, 4=IV", "description": "AJCC pathologic stage"},
    {"name": "t_stage", "role": "predictor", "type": "ordinal", "coding": "0-4", "description": "Pathologic T category"},
    {"name": "n_stage", "role": "predictor", "type": "ordinal", "coding": "0-3", "description": "Pathologic N category"},
    {"name": "m_stage", "role": "predictor", "type": "binary", "coding": "0=M0, 1=M1", "description": "Pathologic M category"},
]


# Function guide: _first_diagnosis_value is responsible for perform diagnosis value.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _first_diagnosis_value(diagnoses: list[dict], key: str):
    for diagnosis in diagnoses or []:
        value = diagnosis.get(key)
        if value is not None:
            return value
    return None


# Function guide: _nonnegative_number is responsible for perform number.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _nonnegative_number(value):
    """Convert a source numeric field without accepting invalid values."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) and number >= 0 else None


# Function guide: download_tcga_luad_records is responsible for download or load cached clinical records.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def download_tcga_luad_records(cache_path: str | Path) -> list[dict]:
    """Download TCGA-LUAD case records or load the local source cache."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        print(f"  GDC cache: {cache_path}", flush=True)
        return json.loads(cache_path.read_text(encoding="utf-8"))

    print("  Downloading TCGA-LUAD from NCI GDC ...", flush=True)
    filters = json.dumps({
        "op": "in",
        "content": {"field": "project.project_id", "value": ["TCGA-LUAD"]},
    })
    records, offset = [], 0
    while True:
        response = requests.get(
            GDC_URL,
            params={
                "filters": filters,
                "fields": GDC_FIELDS,
                "size": "500",
                "from": str(offset),
                "format": "JSON",
            },
            timeout=120,
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        page = response.json()["data"]
        records.extend(page["hits"])
        offset += len(page["hits"])
        if offset >= page["pagination"]["total"]:
            break

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(records), encoding="utf-8")
    print(f"  Downloaded {len(records)} patients", flush=True)
    return records


# Function guide: clean_tcga_luad_records is responsible for normalize clinical records into the model table.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def clean_tcga_luad_records(records: list[dict]) -> pd.DataFrame:
    """Standardize source records and construct the survival outcome.

    This stage performs no train/test-dependent imputation, scaling, or
    functional-form selection. Those operations belong to stage 02.
    """
    rows = []
    seen_case_ids = set()
    excluded = {
        "duplicate_case": 0,
        "unknown_vital_status": 0,
        "invalid_survival_time": 0,
    }
    for record in records:
        case_id = record.get("case_id") or record.get("id")
        if case_id in seen_case_ids:
            excluded["duplicate_case"] += 1
            continue
        if case_id is not None:
            seen_case_ids.add(case_id)

        demographic = record.get("demographic") or {}
        diagnoses = record.get("diagnoses") or []
        vital_status = str(demographic.get("vital_status") or "").strip().upper()
        if vital_status not in {"ALIVE", "DEAD"}:
            excluded["unknown_vital_status"] += 1
            continue

        event = int(vital_status == "DEAD")
        if event:
            survival_days = _nonnegative_number(demographic.get("days_to_death"))
            if survival_days is None:
                survival_days = _nonnegative_number(
                    _first_diagnosis_value(diagnoses, "days_to_death")
                )
        else:
            follow_up = [
                _nonnegative_number(diagnosis.get("days_to_last_follow_up"))
                for diagnosis in diagnoses
            ]
            survival_days = max(
                (value for value in follow_up if value is not None),
                default=None,
            )
        if survival_days is None or survival_days <= 0:
            excluded["invalid_survival_time"] += 1
            continue

        stage_keys = (
            "ajcc_pathologic_stage", "ajcc_pathologic_t",
            "ajcc_pathologic_n", "ajcc_pathologic_m",
        )
        baseline = max(
            diagnoses,
            key=lambda diagnosis: sum(
                diagnosis.get(key) is not None for key in stage_keys
            ),
            default={},
        )
        age_days = _nonnegative_number(baseline.get("age_at_diagnosis"))
        age = age_days / 365.25 if age_days is not None else None
        if age is not None and not 18 <= age <= 100:
            age = None
        m_value = str(
            baseline.get("ajcc_pathologic_m") or ""
        ).strip().upper()
        rows.append({
            "case_id": case_id,
            TIME_COLUMN: survival_days / DAYS_PER_MONTH,
            EVENT_COLUMN: event,
            "age": age,
            "stage": STAGE_MAP.get(
                str(baseline.get("ajcc_pathologic_stage") or "").strip()
            ),
            "t_stage": T_STAGE_MAP.get(
                str(baseline.get("ajcc_pathologic_t") or "").strip()
            ),
            "n_stage": N_STAGE_MAP.get(
                str(baseline.get("ajcc_pathologic_n") or "").strip()
            ),
            "m_stage": (
                1.0 if m_value.startswith("M1")
                else 0.0 if m_value.startswith("M0")
                else None
            ),
        })

    cleaned = pd.DataFrame(rows)
    missing_counts = {
        column: int(cleaned[column].isna().sum())
        for column in cleaned.columns
    } if not cleaned.empty else {}
    audit = {
        "source_rows": len(records),
        "retained_rows": len(cleaned),
        "excluded_rows": excluded,
        "outcome_definition": "Overall survival from diagnosis to death or last follow-up.",
        "event_definition": "1=death; 0=alive at last follow-up and right-censored.",
        "missing_counts": missing_counts,
        "missing_rates": {
            column: count / len(cleaned) if len(cleaned) else 0.0
            for column, count in missing_counts.items()
        },
        "missing_handling": "Retain missing predictor values for training-only imputation in stage 02.",
        "duplicate_policy": "Keep the first record for each case identifier.",
    }
    cleaned.attrs["cleaning_audit"] = audit
    return cleaned.reset_index(drop=True)


# Function guide: write_cleaning_outputs is responsible for write cleaning outputs.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def write_cleaning_outputs(cleaned: pd.DataFrame, output_dir: str | Path) -> None:
    """Write the cleaned table, data dictionary, and audit report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_dir / "tcga_luad_cleaned.csv", index=False)
    (output_dir / "data_dictionary.json").write_text(
        json.dumps(DATA_DICTIONARY, indent=2), encoding="utf-8"
    )
    (output_dir / "data_audit.json").write_text(
        json.dumps(cleaned.attrs.get("cleaning_audit", {}), indent=2),
        encoding="utf-8",
    )


# Function guide: main is responsible for run the module main workflow.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "03_model_training_and_evaluation" / ".cache" / "training",
    )
    parser.add_argument(
        "--source-cache",
        type=Path,
        help="Optional JSON cache path for the raw GDC response.",
    )
    args = parser.parse_args()
    source_cache = args.source_cache or args.output_dir / "tcga_luad_gdc.json"
    records = download_tcga_luad_records(source_cache)
    cleaned = clean_tcga_luad_records(records)
    write_cleaning_outputs(cleaned, args.output_dir)
    print(
        f"Saved cleaned TCGA-LUAD data with {len(cleaned)} patients to {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
