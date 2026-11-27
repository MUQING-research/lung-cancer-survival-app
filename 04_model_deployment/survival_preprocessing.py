# Module guide:
# - Role: Define the deployed survival Shiny application.
# - Workflow: Load a precomputed bundle, validate clinical inputs, estimate survival, and render diagnostics.
# - Design note: The deployed process must not depend on raw training feature matrices.
"""Training-only preprocessing for censored survival outcomes.

This module is the bridge between the cleaned clinical table and the two
deployed survival models.  It defines the raw predictor contract, constructs a
scikit-survival event-time target, and ensures that imputation and
functional-form decisions are fitted on training rows before they are applied
to the held-out rows or to a new patient.
"""

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
    """Build the structured event-time target used by scikit-survival.

    ``event`` is true only for recorded deaths; patients alive at last
    follow-up remain right-censored.  Keeping the named time and event fields
    makes the target self-describing for concordance, IPCW Brier, and dynamic
    AUC calculations.
    """
    # Convert the event column explicitly to boolean because the source table
    # may store it as integer 0/1 values.
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
    """Fit or apply the clinical preprocessor without leaking test information.

    With ``fit=True``, imputation values, category levels, and functional-form
    choices are estimated from ``data`` and its event-time outcome.  With
    ``fit=False``, a previously fitted object must be supplied and is applied
    unchanged.  The returned DataFrame contains model-ready columns only; the
    duration and event columns remain in the original clinical table.
    """
    check_feature_schema(data)
    # Select the exact raw feature schema before fitting or transforming.  This
    # prevents outcome columns and identifiers from entering either model.
    raw_features = data[FEATURE_COLUMNS].copy().astype(float)
    if fit:
        # The outcome is used only for Cox functional-form diagnostics.  The
        # preprocessor itself still estimates missing-value replacements from
        # the current training features.
        preprocessor = sc.ClinicalPreprocessor().fit(
            raw_features, data[[sc.TIME_COL, sc.EVENT_COL]]
        )
    if preprocessor is None:
        raise ValueError("A fitted preprocessor is required when fit=False.")
    # Transform with frozen training statistics and return the fitted object so
    # the caller can reuse it for the test split and future app inputs.
    return preprocessor.transform(raw_features), preprocessor


# Function guide: split_and_preprocess is responsible for split and preprocess.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def split_and_preprocess(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, sc.ClinicalPreprocessor]:
    """Create train/test design matrices using training-only preprocessing.

    This helper makes the leakage boundary explicit: the first call fits the
    preprocessor, and the second call only transforms the test data.  The same
    fitted object is returned for bundle persistence and deployment scoring.
    """
    # Never call ``fit=True`` on the test table.  Doing so would let its missing
    # value frequencies or functional-form evidence influence model inputs.
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
    """Return an auditable record of training-set preprocessing choices.

    The decision log records variable type, missingness, imputation value,
    selected functional form, reference levels, spline cut points, and the
    diagnostic results that led to the choice.  It is saved with the bundle so
    a later user can reproduce the model-input contract without raw patient
    feature rows.
    """
    decisions = {}
    for index, name in enumerate(FEATURE_COLUMNS):
        # All values below are sourced from the training table or fitted
        # preprocessor; no test-set statistic is copied into the log.
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
                else "restricted cubic spline basis" if name == "age"
                else "integer code"
            ),
            "reference_level": preprocessor.categories_.get(name, [None])[0],
            "training_levels": preprocessor.categories_.get(name),
            "cut_points": (
                preprocessor.age_rcs_.knots_.tolist()
                if name == "age" and form == "rcs" else None
            ),
            "form_checks": [
                result for result in preprocessor.form_checks_
                if result.get("variable") == name
            ],
            "missing_indicator": "not included; model fitting uses complete cases only",
        }
    return decisions


# Function guide: check_feature_schema is responsible for check feature schema.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def check_feature_schema(data: pd.DataFrame) -> None:
    """Reject invalid predictor schemas before model fitting or scoring.

    Additional columns are tolerated so source identifiers can remain in a
    cleaned table, but every required predictor must be present, finite, and
    non-missing. The deployed model is intentionally restricted to complete
    clinical profiles.
    """
    # Check the required names first so the error identifies the data-contract
    # problem instead of failing later inside a numerical estimator.
    missing = [column for column in FEATURE_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing clinical predictor columns: {missing}")
    values = data[FEATURE_COLUMNS].to_numpy(dtype=float)
    if np.isnan(values).any() or not np.isfinite(values).all():
        raise ValueError("Clinical predictors must be finite and complete.")
