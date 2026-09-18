# Module guide:
# - Role: Define the deployed survival Shiny application.
# - Workflow: Load a precomputed bundle, validate clinical inputs, estimate survival, and render diagnostics.
# - Design note: The deployed process must not depend on raw training feature matrices.
"""Training-only preprocessing for censored survival outcomes."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sksurv.util import Surv

import survival_core as sc


FEATURE_COLUMNS = ["age", "stage", "t_stage", "n_stage", "m_stage"]
FEATURE_DISPLAY = {
    "age": "Age (years)",
    "stage": "AJCC Pathologic Stage (I-IV)",
    "t_stage": "Pathologic T Category (T0-T4)",
    "n_stage": "Pathologic N Category (N0-N3)",
    "m_stage": "Pathologic M Category (M0/M1)",
}
VARIABLE_TYPES = {
    "age": "continuous",
    "stage": "ordinal",
    "t_stage": "ordinal",
    "n_stage": "ordinal",
    "m_stage": "binary",
}


# Function guide: build_survival_target is responsible for build survival target.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def build_survival_target(data: pd.DataFrame):
    """Build a structured event-time target for scikit-survival."""
    return Surv.from_arrays(
        event=data[sc.EVENT_COL].astype(bool).to_numpy(),
        time=data[sc.TIME_COL].to_numpy(),
        name_event=sc.EVENT_COL,
        name_time=sc.TIME_COL,
    )


# Function guide: transform_feature_matrix is responsible for transform feature matrix.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def transform_feature_matrix(
    data: pd.DataFrame,
    preprocessor: sc.ClinicalPreprocessor | None = None,
    fit: bool = False,
) -> tuple[pd.DataFrame, sc.ClinicalPreprocessor]:
    """Fit on training data or apply an already fitted preprocessing object."""
    raw_features = data[FEATURE_COLUMNS].copy().astype(float)
    if fit:
        preprocessor = sc.ClinicalPreprocessor().fit(
            raw_features, data[[sc.TIME_COL, sc.EVENT_COL]]
        )
    if preprocessor is None:
        raise ValueError("A fitted preprocessor is required when fit=False.")
    return preprocessor.transform(raw_features), preprocessor


# Function guide: split_and_preprocess is responsible for split and preprocess.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def split_and_preprocess(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, sc.ClinicalPreprocessor]:
    """Fit all imputation and functional-form rules on train data only."""
    train_features, preprocessor = transform_feature_matrix(train_data, fit=True)
    test_features, _ = transform_feature_matrix(test_data, preprocessor=preprocessor)
    return train_features, test_features, preprocessor


# Function guide: build_preprocessing_decisions is responsible for build preprocessing decisions.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def build_preprocessing_decisions(
    train_data: pd.DataFrame,
    preprocessor: sc.ClinicalPreprocessor,
) -> dict[str, dict]:
    """Return a machine-readable record of training-set preprocessing choices."""
    decisions = {}
    for index, name in enumerate(FEATURE_COLUMNS):
        train_values = train_data[name]
        form = preprocessor.forms_[name]
        decisions[name] = {
            "variable_type": VARIABLE_TYPES[name],
            "missing_value_rule": "median" if name == "age" else "mode",
            "missing_count_train": int(train_values.isna().sum()),
            "missing_rate_train": float(train_values.isna().mean()),
            "imputation_value": float(preprocessor.statistics_[index]),
            "functional_form": form,
            "encoding": (
                "dummy, drop first" if form == "dummy"
                else "continuous" if name == "age"
                else "integer code"
            ),
            "reference_level": preprocessor.categories_.get(name, [None])[0],
            "training_levels": preprocessor.categories_.get(name),
            "cut_points": (
                preprocessor.age_spline_.bsplines_[0].t.tolist()
                if name == "age" and form == "spline" else None
            ),
            "form_checks": [
                result for result in preprocessor.form_checks_
                if result.get("variable") == name
            ],
        }
    return decisions


# Function guide: check_feature_schema is responsible for check feature schema.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def check_feature_schema(data: pd.DataFrame) -> None:
    """Reject missing or unexpected predictor columns before model fitting."""
    missing = [column for column in FEATURE_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing clinical predictor columns: {missing}")
    values = data[FEATURE_COLUMNS].to_numpy(dtype=float)
    finite_values = values[~np.isnan(values)]
    if not np.isfinite(finite_values).all():
        raise ValueError("Clinical predictors contain non-finite values.")
