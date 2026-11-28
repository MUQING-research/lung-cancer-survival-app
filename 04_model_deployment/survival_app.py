"""
survival_app.py — TCGA-LUAD Lung Adenocarcinoma Overall Survival
================================================================
Cox Proportional Hazards  ×  Log-Logistic Accelerated Failure Time
Clinical data: TCGA-LUAD · N = 509 patients · NCI GDC API
Cell Press visual style · Research & educational use only

Modeling notes
--------------
The application compares a regularized Cox proportional-hazards model with a
parametric accelerated-failure-time model.  Death is the event and patients
alive at last follow-up are right-censored.  A fixed stratified 80/20 split is
created after base cleaning; all imputation, functional-form decisions,
penalizer tuning, and AFT family selection use training data only.  The test
partition is used once for final discrimination and prediction-error metrics.
"""

# Module guide:
# - Role: Define the deployed survival Shiny application.
# - Workflow: Load a precomputed bundle, validate clinical inputs, estimate survival, and render diagnostics.
# - Design note: The deployed process must not depend on raw training feature matrices.
from __future__ import annotations

import hashlib
import html
import json
import os
import sys
import tempfile
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from shiny import App, reactive, render, ui
from shiny.types import SafeException
from sklearn.model_selection import train_test_split
from sksurv.metrics import brier_score as _brier_score

from visualizations import CELL_COLORS
import survival_core as sc
from survival_core import (
    BRAND, EVENT_COL, SEED, TIME_COL,
    build_aft, build_cox, evaluate, test_distributions,
)

warnings.filterwarnings("ignore")

_CACHE_ROOT = Path(os.environ.get(
    "SURVIVAL_CACHE_DIR",
    str(Path(__file__).parent / ".cache"),
))

if os.name == "nt":
    _TMP_ROOT = _CACHE_ROOT / "tmp"
    _TMP_ROOT.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(_TMP_ROOT)

    # Class guide: _SafeTemporaryDirectory is responsible for perform the routine-specific application step.
    # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    class _SafeTemporaryDirectory:
        # Function guide: __init__ is responsible for perform the routine-specific application step.
        # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
        # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
        def __init__(self, suffix=None, prefix=None, dir=None,
                     ignore_cleanup_errors=True):
            self.name = tempfile.mkdtemp(
                suffix=suffix or "",
                prefix=prefix or "tmp",
                dir=dir or str(_TMP_ROOT),
            )

        # Function guide: __enter__ is responsible for perform the routine-specific application step.
        # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
        # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
        def __enter__(self):
            return self.name

        # Function guide: __exit__ is responsible for perform the routine-specific application step.
        # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
        # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
        def __exit__(self, exc_type, exc, tb):
            return False

        # Function guide: cleanup is responsible for perform the routine-specific application step.
        # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
        # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
        def cleanup(self):
            return None

    tempfile.TemporaryDirectory = _SafeTemporaryDirectory

sc.apply_cell_matplotlib_style()

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _source_dir in (
    _REPO_ROOT / "01_data_cleaning",
    _REPO_ROOT / "02_data_preprocessing",
):
    if _source_dir.is_dir():
        sys.path.insert(0, str(_source_dir))

from clean_clinical_data import (  # noqa: E402
    clean_tcga_luad_records,
    download_tcga_luad_records,
)
from survival_preprocessing import (  # noqa: E402
    FEATURE_COLUMNS,
    FEATURE_DISPLAY,
    build_survival_target,
    transform_feature_matrix,
)

_COX_CLR = CELL_COLORS[0]
_AFT_CLR = CELL_COLORS[1]
_MUTED   = BRAND["brown"]
_AXIS_SV = "#334155"     # axis spines
_TICK_SV = "#475569"     # tick marks / tick labels
_LABEL_SV = "#1E293B"    # axis labels
_GRID_SV = "#E2E8F0"     # neutral reference grid colour
_PANEL_SV = BRAND["navy"]  # navy panel titles

# ── 1. Runtime paths and data schema ─────────────────────────────────────────

# On shinyapps.io: /tmp is writable. Locally: fall back to a .cache sibling.
_CACHE_DIR = Path(os.environ.get(
    "SURVIVAL_TRAINING_CACHE_DIR",
    str(Path(__file__).parent / ".cache"),
))
_GDC_CACHE = _CACHE_DIR / "tcga_luad_gdc.json"

FEAT_COLS = FEATURE_COLUMNS
FEAT_DISPLAY = FEATURE_DISPLAY


# Function guide: _download_gdc is responsible for perform the routine-specific application step.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _download_gdc() -> list:
    return download_tcga_luad_records(_GDC_CACHE)


# Function guide: _clean_clinical_records is responsible for perform the routine-specific application step.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _clean_clinical_records(hits: list) -> pd.DataFrame:
    """Delegate base cleaning to stage 01 without data-dependent fitting."""
    return clean_tcga_luad_records(hits)


# Function guide: _build_survival_target is responsible for prepare or evaluate survival model information.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _build_survival_target(df: pd.DataFrame):
    return build_survival_target(df)


# Function guide: _transform_feature_matrix is responsible for perform the routine-specific application step.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _transform_feature_matrix(
    df: pd.DataFrame,
    imputer: sc.ClinicalPreprocessor = None,
    fit: bool = False,
) -> tuple[pd.DataFrame, sc.ClinicalPreprocessor]:
    return transform_feature_matrix(df, preprocessor=imputer, fit=fit)


_BUNDLE_VERSION = 9
N_IMPUTATIONS = 5


# Function guide: _stage_counts is responsible for perform the routine-specific application step.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _stage_counts(df: pd.DataFrame) -> dict[str, int]:
    return {str(i): int((df["stage"] == i).sum()) for i in range(1, 5)}


# Function guide: _null_brier_curve is responsible for prepare or evaluate survival model information.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _null_brier_curve(df_train: pd.DataFrame, y_train, y_test,
                      n_test: int, times: list | np.ndarray) -> dict:
    """Build a Kaplan-Meier null reference for Brier-score interpretation.

    The null model assigns every test patient the same training-set Kaplan-
    Meier survival probability at each horizon.  It is a no-covariate
    baseline, not a fitted competing model, and is used only to contextualize
    the covariate-adjusted survival predictions.
    """
    times_arr = np.array(times, dtype=float)
    if len(times_arr) == 0:
        return {"times": [], "values": []}
    try:
        # Estimate the reference survival curve from training rows only.
        km_ref = KaplanMeierFitter().fit(df_train[TIME_COL], df_train[EVENT_COL])
        ref_sf = np.array([km_ref.predict(t) for t in times_arr], dtype=float)
        # Repeat the same curve for every test row, then evaluate it with the
        # same IPCW censoring treatment used for the actual model predictions.
        null_m = np.tile(ref_sf, (n_test, 1))
        _, null_bs = _brier_score(y_train, sc.administrative_censor(y_test, times_arr), null_m, times_arr)
        return {"times": times_arr.tolist(), "values": np.asarray(null_bs).tolist()}
    except Exception:
        return {"times": [], "values": []}


# Function guide: _sanitized_bundle is responsible for load, validate, or save deployment bundle state.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _sanitized_bundle(raw: dict) -> tuple[dict, bool]:
    """Keep only deployment-safe survival artifacts and remove fit caches.

    Fitted model parameters, preprocessing statistics, outcomes, metrics, and
    aggregate diagnostics are sufficient for deployment.  Raw training/test
    feature tables and optimizer caches are not needed for prediction and are
    removed before the bundle is written.  The boolean result tells startup
    whether the on-disk artifact was changed and should be rewritten.
    """
    df_train = raw.get("df_train")
    df_all = raw.get("df_all")
    y_train = raw["y_train"]
    y_test = raw["y_test"]
    n_train = int(raw.get("n_train", len(y_train)))
    n_test = int(raw.get("n_test", len(y_test)))
    # Derive aggregate stage counts before dropping any legacy row-level table.
    stage_source = df_all if isinstance(df_all, pd.DataFrame) else df_train
    stage_counts = raw.get("stage_counts")
    if stage_counts is None and isinstance(stage_source, pd.DataFrame):
        stage_counts = _stage_counts(stage_source)

    res_cox = raw["res_cox"]
    null_brier = raw.get("null_brier")
    if null_brier is None and isinstance(df_train, pd.DataFrame):
        null_brier = _null_brier_curve(
            df_train, y_train, y_test, n_test, res_cox.get("times", EVAL_TIMES)
        )

    # This allowlist is the deployment boundary.  In particular, neither
    # df_train nor df_all is copied into the cleaned artifact.
    keep = [
        "y_train", "y_test", "n_total", "ev_rate", "med_fu",
        "dist", "km_train", "cox", "aft", "res_cox", "res_aft",
        "tr_cox", "tr_aft", "imp_statistics", "feat_cols",
        "n_imputations", "n_test_evaluable",
        "preprocessing_decisions", "provenance", "diagnostics",
        "preprocessor",
    ]
    clean = {k: raw[k] for k in keep if k in raw}
    clean.update(
        bundle_version=_BUNDLE_VERSION,
        n_train=n_train,
        n_test=n_test,
        n_test_evaluable=int(raw.get("n_test_evaluable", n_test)),
        stage_counts=stage_counts or {str(i): 0 for i in range(1, 5)},
        null_brier=null_brier or {"times": [], "values": []},
    )
    changed = set(raw) != set(clean) or raw.get("bundle_version") != _BUNDLE_VERSION
    # Remove fit-only caches. Prediction requires coefficients, baseline
    # hazards, preprocessing statistics, and feature schema; no individual
    # training feature matrix is needed.
    for name in ("cox", "aft"):
        model = clean[name]
        model_objects = [model]
        model_objects.extend(getattr(model, "models", []))
        owners = []
        for fitted in model_objects:
            owners.extend([fitted, vars(fitted).get("_model")])
        for owner in owners:
            if owner is None:
                continue
            for key in ("_training_data", "_training_df", "_X", "_Xs", "_norm_X",
                        "_predicted_median", "_predicted_partial_hazards_",
                        "_neg_likelihood", "_neg_likelihood_with_penalty_function"):
                if key in vars(owner):
                    del vars(owner)[key]
                    changed = True
    return clean, changed


# Function guide: _save_bundle is responsible for load, validate, or save deployment bundle state.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _save_bundle(bundle: dict, path: Path) -> None:
    """Persist a deployment bundle after creating its parent directory."""
    import pickle
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as _f:
        pickle.dump(bundle, _f)


# ── 2. Startup — load bundle if available, else download + train from scratch

_BUNDLE_PATH = Path(__file__).parent / "tcga_luad_survival_model_bundle.pkl"
EVAL_TIMES   = np.array([12., 24., 36., 48., 60.])


# Function guide: _startup_from_bundle is responsible for load, validate, or save deployment bundle state.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _startup_from_bundle(path: Path) -> dict:
    """Load an existing bundle without downloading or refitting data."""
    import pickle
    print(f"  Loading bundle: {path}", flush=True)
    with open(path, "rb") as f:
        b = pickle.load(f)
    print(f"  N={b['n_total']}  events={b['ev_rate']:.0%}  "
          f"median FU={b['med_fu']:.0f}m", flush=True)
    return b


# Function guide: _startup_from_scratch is responsible for perform the routine-specific application step.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _startup_from_scratch() -> dict:
    """Download, clean, split, fit, evaluate, and package both models.

    This is the explicit rebuild path.  Normal deployment loads the saved
    bundle instead, while the rebuild path is useful for a reproducible model
    refresh or an offline audit.  The sequence is intentionally visible here:
    base cleaning -> stratified split -> training-only preprocessing ->
    distribution/model fitting -> test evaluation -> bundle sanitization.
    """
    # Base cleaning standardizes source records without fitting any model
    # parameters or using outcome-aware statistics.
    hits      = _download_gdc()
    df        = _clean_clinical_records(hits)
    complete_case_count = int(df[FEAT_COLS].notna().all(axis=1).sum())
    n_total   = len(df)
    ev_rate   = df[EVENT_COL].mean()
    med_fu    = df[TIME_COL].median()
    print(
        f"  N={n_total}  complete_cases={complete_case_count}  "
        f"events={ev_rate:.0%}  median FU={med_fu:.0f}m", flush=True)

    # Split before imputation, functional-form screening, or model fitting so
    # every learned preprocessing value comes from the training partition.
    df_train, df_test = train_test_split(
        df, test_size=0.20, random_state=SEED, stratify=df[EVENT_COL],
    )
    df_train = df_train.reset_index(drop=True)
    df_test  = df_test.reset_index(drop=True)

    # The held-out test partition is never used to fit preprocessing.  The
    # training-fitted imputer is applied to every test row, which is the
    # standard leakage-safe deployment pattern for models that cannot consume
    # missing values directly.
    test_missing = df_test[FEAT_COLS].isna().any(axis=1)
    df_test_eval = df_test.copy()
    test_missing_rows = int(test_missing.sum())
    print(
        f"  Test rows={len(df_test)}  evaluated={len(df_test_eval)}  "
        f"rows_transformed_with_train_imputer={test_missing_rows}", flush=True)

    # Structured targets retain the event/time field names expected by
    # scikit-survival's IPCW and concordance implementations.
    y_train = _build_survival_target(df_train)
    y_test  = _build_survival_target(df_test_eval)

    # Fit five posterior-sampling chained-equation imputers on the training
    # rows only.  Every imputed dataset receives its own functional-form
    # preprocessor, preserving imputation and model-form uncertainty.
    imp = sc.MultipleClinicalPreprocessor(
        n_imputations=N_IMPUTATIONS, random_state=SEED,
    ).fit(df_train[FEAT_COLS], df_train[[TIME_COL, EVENT_COL]])
    ft_train_list = imp.transform_multiple(df_train[FEAT_COLS])
    ft_test_list = imp.transform_multiple(df_test_eval[FEAT_COLS])

    print("  Distribution fitting ...", flush=True)
    # Compare marginal survival families using training observations only.
    # The selected family is then used for the covariate-adjusted AFT fit.
    dist     = test_distributions(df_train)
    km_train = KaplanMeierFitter().fit(
        df_train[TIME_COL], df_train[EVENT_COL], label="Kaplan-Meier"
    )

    print("  Training Cox PH across imputations ...", flush=True)
    # Select one common penalty with fold-local MICE preprocessing, then refit
    # the Cox model on each completed training dataset.
    cox_first = build_cox(
        df_train, feat_df=ft_train_list[0], raw_feat_df=df_train[FEAT_COLS],
        fold_imputation="mice",
    )
    cox_models = [cox_first]
    for features in ft_train_list[1:]:
        cox_models.append(build_cox(
            df_train, penalizer=cox_first._penalizer_used, feat_df=features,
        ))
    cox = sc.SurvivalEnsemble(cox_models, "cox")
    print(f"    penalizer={cox._penalizer_used}  train C={cox.concordance_index_:.3f}",
          flush=True)

    print(f"  Training {dist['best']} AFT across imputations ...", flush=True)
    # Fit one AFT model per completed training dataset and pool predictions.
    aft_models = [
        build_aft(df_train, aft_class=dist["aft_class"], feat_df=features)
        for features in ft_train_list
    ]
    aft = sc.SurvivalEnsemble(aft_models, "aft")

    eval_times = sc._clip_times(EVAL_TIMES, y_train, y_test)
    print("  Evaluating test set (n_boot=150) ...", flush=True)
    # Evaluate the untouched test partition with bootstrap C-index intervals,
    # IPCW integrated Brier score, and time-dependent AUC.
    res_cox = evaluate(cox, "cox", df_test_eval, y_train, y_test,
                       eval_times, feat_df_test=ft_test_list, n_boot=150)
    res_aft = evaluate(aft, "aft", df_test_eval, y_train, y_test,
                       eval_times, feat_df_test=ft_test_list, n_boot=150)
    print(
        f"  Cox  C={res_cox['c_index']:.3f} [{res_cox['ci_lo']:.3f},{res_cox['ci_hi']:.3f}]"
        f"  IBS={res_cox['ibs']:.4f}", flush=True,
    )
    print(
        f"  AFT  C={res_aft['c_index']:.3f} [{res_aft['ci_lo']:.3f},{res_aft['ci_hi']:.3f}]"
        f"  IBS={res_aft['ibs']:.4f}", flush=True,
    )

    print("  Computing training-set metrics ...", flush=True)
    # Apparent train metrics are kept beside test metrics to expose possible
    # overfitting; they are not presented as independent validation.
    tr_cox = sc.evaluate_train(cox, "cox", df_train, y_train,
                               eval_times, feat_df_train=ft_train_list)
    tr_aft = sc.evaluate_train(aft, "aft", df_train, y_train,
                               eval_times, feat_df_train=ft_train_list)
    print(f"  Cox  train C={tr_cox['c_index']:.3f}  train IBS={tr_cox['ibs']:.4f}",
          flush=True)
    print(f"  AFT  train C={tr_aft['c_index']:.3f}  train IBS={tr_aft['ibs']:.4f}",
          flush=True)

    # Build a no-covariate Brier reference for the performance panel.
    null_brier = _null_brier_curve(df_train, y_train, y_test, len(df_test_eval),
                                   res_cox.get("times", EVAL_TIMES))

    for label, result in (("Cox test", res_cox), ("AFT test", res_aft), ("Cox train", tr_cox), ("AFT train", tr_aft)):
        if not all(np.isfinite(result[key]) for key in ("c_index", "ibs")) or result.get("metric_errors"):
            raise RuntimeError(f"Invalid {label} evaluation: {result.get('metric_errors', {})}")

    # Export a machine-readable decision log and provenance record so every
    # deployed prediction can be traced back to the training specification.
    decisions, diagnostics = sc.audit_clinical_preprocessing(
        df, df_train, ft_train_list[0], cox_first, imp,
    )
    # Save inspectable continuous-form and categorical-encoding figures using
    # the outer training partition only.  Only continuous variables receive a
    # linearity diagnostic; dummy-coded categories do not.
    audit_dir = Path(os.environ.get(
        "SURVIVAL_AUDIT_DIR",
        str(Path(__file__).parent / ".cache" / "audit"),
    ))
    audit_dir.mkdir(parents=True, exist_ok=True)
    # Render the article-style adjusted RCS HR curve and annotate its overall
    # and nonlinear P values in all five completed training datasets.  No
    # raw-age-squared or secondary diagnostic figure is generated here.
    feature_form_figures = sc.save_feature_form_diagnostics(
        df_train,
        imp,
        audit_dir,
        age_only=True,
    )
    diagnostics["feature_form_figures"] = feature_form_figures
    complete_case_sensitivity_n = int(df[FEAT_COLS].notna().all(axis=1).sum())
    provenance = {
        "seed": SEED, "split": "80/20 event-stratified", "n_train": len(df_train), "n_test": len(df_test),
        "n_test_evaluable": len(df_test_eval), "test_non_evaluable_missing": 0,
        "test_rows_with_missing_predictors": test_missing_rows,
        "train_events": int(df_train[EVENT_COL].sum()), "test_events": int(df_test_eval[EVENT_COL].sum()),
        "cleaning": {
            **df.attrs.get("cleaning_audit", {}),
            "complete_case_sensitivity_n": complete_case_sensitivity_n,
            "missing_predictor_rule": "Fit MICE within the training split; apply the fitted transformer to validation, held-out test, and deployment inputs without refitting.",
            "outcome_exclusion_rule": "Exclude only duplicate cases, unknown vital status, or invalid/non-positive survival time.",
        },
        "missing_data": {
            "primary_method": "Multiple imputation by chained equations with posterior sampling",
            "n_imputations": N_IMPUTATIONS,
            "fit_scope": "Imputer fitted on training partition only; applied unchanged to test and deployment inputs",
            "missing_indicator": "Used for audit of missingness patterns; not included as a prediction feature.",
            "sensitivity_analysis": "Complete-case analysis retained as a sensitivity comparison; MNAR delta sensitivity remains recommended.",
        },
        "source_sha256": hashlib.sha256(json.dumps(hits, sort_keys=True).encode()).hexdigest(),
        "cox_cv": "Five event-stratified folds; imputation, article-style RCS diagnostics, selected age form and dummy levels refitted in each training fold",
        "evaluation_times": eval_times.tolist(),
        "versions": {"lifelines": sc.lifelines.__version__, "scikit_survival": sc.sksurv.__version__, "scikit_learn": sc.sklearn.__version__},
    }
    df.to_csv(audit_dir / "tcga_luad_cleaned.csv", index=False)
    ft_train_list[0].to_csv(audit_dir / "X_train_design.csv", index=False)
    ft_test_list[0].to_csv(audit_dir / "X_test_design.csv", index=False)
    (audit_dir / "data_dictionary.json").write_text(json.dumps(decisions, indent=2), encoding="utf-8")
    (Path(__file__).parent / "preprocessing_decisions.json").write_text(
        json.dumps({"variables": decisions, "provenance": provenance, "diagnostics": diagnostics}, indent=2, allow_nan=False), encoding="utf-8",
    )
    bundle = dict(
        bundle_version=_BUNDLE_VERSION,
        y_train=y_train, y_test=y_test,
        n_total=n_total, ev_rate=ev_rate, med_fu=med_fu,
        n_train=len(df_train), n_test=len(df_test),
        n_test_evaluable=len(df_test_eval),
        stage_counts=_stage_counts(df),
        dist=dist, km_train=km_train,
        cox=cox, aft=aft,
        res_cox=res_cox, res_aft=res_aft,
        tr_cox=tr_cox, tr_aft=tr_aft,
        null_brier=null_brier,
        imp_statistics=imp.statistics_.tolist(), feat_cols=FEAT_COLS,
        n_imputations=N_IMPUTATIONS,
        preprocessing_decisions=decisions, provenance=provenance, diagnostics=diagnostics,
        preprocessor=imp,
    )
    return _sanitized_bundle(bundle)[0]


print("=" * 56, flush=True)
print("TCGA-LUAD Survival App — initialising", flush=True)
print("=" * 56, flush=True)

if _BUNDLE_PATH.exists() and os.environ.get("SURVIVAL_REBUILD_BUNDLE") != "1":
    # Production path: load the precomputed artifact and avoid network access
    # or model fitting during ordinary app startup.
    _raw_bundle = _startup_from_bundle(_BUNDLE_PATH)
    if _raw_bundle.get("bundle_version") != _BUNDLE_VERSION:
        raise RuntimeError("Outdated survival bundle. Run 03_model_training_and_evaluation/train_and_export_model_bundle.py before starting the app.")
    _B, _changed = _sanitized_bundle(_raw_bundle)
    if _changed:
        _save_bundle(_B, _BUNDLE_PATH)
        print(f"  Bundle sanitized: {_BUNDLE_PATH}", flush=True)
else:
    # Rebuild path used when the bundle is missing or explicitly invalidated.
    _B = _startup_from_scratch()
    _save_bundle(_B, _BUNDLE_PATH)
    print(f"  Bundle saved: {_BUNDLE_PATH}", flush=True)

# ── Unpack bundle ─────────────────────────────────────────────────────────────
# The fitted transformer stores training statistics, spline knots, and category
# reference levels.  It is reused for every patient prediction below.
_IMP = _B["preprocessor"]

N_TOTAL  = _B["n_total"]
EV_RATE  = _B["ev_rate"]
MED_FU   = _B["med_fu"]
N_TRAIN  = _B["n_train"]
N_TEST   = _B["n_test"]
N_TEST_EVAL = _B.get("n_test_evaluable", N_TEST)
STAGE_COUNTS = _B.get("stage_counts", {str(i): 0 for i in range(1, 5)})
NULL_BRIER = _B.get("null_brier", {"times": [], "values": []})

DIST     = _B["dist"]
KM_TRAIN = _B["km_train"]
COX      = _B["cox"]
AFT      = _B["aft"]
RES_COX  = _B["res_cox"]
RES_AFT  = _B["res_aft"]
TR_COX   = _B.get("tr_cox", {})
TR_AFT   = _B.get("tr_aft", {})

print("  Ready.", flush=True)


# ── 4. Plotting helpers ──────────────────────────────────────────────────────

_CURVE_T = np.linspace(0.1, 72, 300)
_KEY_T   = np.array([6., 12., 18., 24., 36., 48., 60.])


# Function guide: _cell_ax is responsible for perform the routine-specific application step.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _cell_ax(fig, ax, grid=False):
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for sp in ("top", "right", "bottom", "left"):
        ax.spines[sp].set_visible(True)
        ax.spines[sp].set_color(_AXIS_SV)
        ax.spines[sp].set_linewidth(0.8)
    ax.tick_params(colors=_TICK_SV, direction="out", length=2.6, width=0.7, labelsize=7.5)
    ax.xaxis.label.set_color(_LABEL_SV)
    ax.yaxis.label.set_color(_LABEL_SV)
    ax.xaxis.label.set_size(8.5)
    ax.yaxis.label.set_size(8.5)
    ax.title.set_color(_PANEL_SV)
    ax.title.set_size(9.0)
    if grid:
        ax.grid(True, axis="y", color=_GRID_SV, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
    else:
        ax.grid(False)


# Function guide: _predict_curve is responsible for perform the routine-specific application step.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _predict_curve(model, feat_df: pd.DataFrame) -> np.ndarray:
    """Return one patient's survival probabilities on the shared plot grid.

    Lifelines versions differ in whether ``predict_survival_function`` accepts
    a ``times`` argument.  The direct call is preferred; the fallback obtains
    the model's internal event-time curve and interpolates it onto the common
    grid.  Clipping protects the UI from tiny floating-point excursions beyond
    the probability interval.
    """
    try:
        # Newer lifelines can evaluate directly at the requested horizons.
        sf    = model.predict_survival_function(feat_df, times=_CURVE_T)
        probs = sf.iloc[:, 0].values
    except TypeError:
        # Older versions return their internal time grid; interpolate the one
        # patient column without refitting or changing the model.
        sf    = model.predict_survival_function(feat_df)
        s     = sf.iloc[:, 0]
        probs = np.interp(_CURVE_T, s.index.values, s.values, left=1.0)
    return probs.clip(0.0, 1.0)


# Function guide: _survival_at_times is responsible for prepare or evaluate survival model information.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _survival_at_times(model, times: np.ndarray) -> np.ndarray:
    """Return clipped model survival probabilities at selected horizons."""
    vals = model.survival_function_at_times(times)
    arr = np.asarray(vals, dtype=float).reshape(-1)
    return np.clip(arr, 0.0, 1.0)


# Function guide: _survival_function_frame is responsible for prepare or evaluate survival model information.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _survival_function_frame(model):
    """Expose a fitted model's baseline survival curve for plotting."""
    sf = getattr(model, "survival_function_", None)
    if sf is None or sf.empty:
        return None
    vals = sf.iloc[:, 0].astype(float)
    return vals.index.to_numpy(dtype=float), vals.to_numpy(dtype=float)


import visualizations as charts

_CINDEX_SRC = charts.png(charts.cindex_figure(globals()))
_AUC_SRC = charts.png(charts.auc_figure(globals()))
_DIST_SRC = charts.png(charts.distribution_figure(globals()))
print("  Static figures rendered.", flush=True)


# ── 4. CSS (Cell Press style) ────────────────────────────────────────────────

# Keep the dashboard theme in one deployable stylesheet.
_CSS = (Path(__file__).parent / "compact_theme.css").read_text(encoding="utf-8")

_TEXT_SCALE_JS = """
(() => {
  const storageKey = "medictio-text-scale";
  const allowed = new Set(["1", "1.125", "1.25"]);
  const applyScale = (value) => {
    const scale = allowed.has(value) ? value : "1";
    document.documentElement.style.fontSize = `${Number(scale) * 100}%`;
    return scale;
  };
  document.addEventListener("DOMContentLoaded", () => {
    const control = document.getElementById("text-scale");
    if (!control) return;
    let saved = "1";
    try { saved = localStorage.getItem(storageKey) || saved; } catch (_) {}
    control.value = applyScale(saved);
    control.addEventListener("change", () => {
      const value = applyScale(control.value);
      try { localStorage.setItem(storageKey, value); } catch (_) {}
    });
  });
})();
"""


_stage_lbl = {"1": "I", "2": "II", "3": "III", "4": "IV"}
_t_lbl = {str(i): f"T{i}" for i in range(5)}
_n_lbl = {str(i): f"N{i}" for i in range(4)}
_m_lbl = {"0": "M0", "1": "M1"}
_MISSING_OPTION = {"missing": "Unknown"}


def _section_head(title: str, copy: str) -> ui.Tag:
    return ui.tags.div(ui.tags.h2(title), ui.tags.p(copy), class_="section-head")


def _card(title: str, *content: ui.Tag, subtitle: str | None = None) -> ui.Tag:
    heading = ui.tags.div(
        ui.tags.h3(title), ui.tags.p(subtitle) if subtitle else None,
        class_="card-heading",
    )
    return ui.tags.section(heading, *content, class_="dashboard-card")


def _stat_tile(label: str, value: str, detail: str) -> ui.Tag:
    return ui.tags.div(
        ui.tags.span(label, class_="stat-label"),
        ui.tags.strong(value, class_="stat-value"),
        ui.tags.span(detail, class_="stat-detail"),
        class_="stat-tile",
    )


def _stage_table() -> ui.Tag:
    counts = [(f"Stage {label}", int(STAGE_COUNTS.get(code, 0)))
              for code, label in _stage_lbl.items()]
    counts.append(("Unknown", N_TOTAL - sum(count for _, count in counts)))
    rows = [
        ui.tags.tr(
            ui.tags.th(label, scope="row"),
            ui.tags.td(f"{count:,}", class_="num"),
            ui.tags.td(f"{count / N_TOTAL:.1%}" if N_TOTAL else "N/A", class_="num"),
        )
        for label, count in counts
    ]
    return ui.tags.div(
        ui.tags.table(
            ui.tags.thead(ui.tags.tr(
                ui.tags.th("Pathologic stage", scope="col"),
                ui.tags.th("Patients", scope="col"),
                ui.tags.th("Share of cohort", scope="col"),
            )),
            ui.tags.tbody(*rows),
            class_="mtbl",
        ),
        class_="table-frame",
    )


def _detail_panel(label: str, *content: ui.Tag) -> ui.Tag:
    return ui.tags.details(ui.tags.summary(label), *content, class_="detail-panel")


def _chart(output_id: str, extra_class: str = "") -> ui.Tag:
    return ui.tags.div(ui.output_ui(output_id), class_=f"chart-frame {extra_class}".strip())


def _input_panel() -> ui.Tag:
    return ui.tags.aside(
        ui.tags.div(
            ui.tags.h3("Patient profile"),
            ui.tags.p("Enter age at diagnosis and pathologic stage."),
            class_="card-heading",
        ),
        ui.tags.div("Patient and disease stage", class_="input-group-title"),
        ui.tags.div(
            ui.input_numeric("age", "Age (years)", value=65, min=18, max=100, step=1),
            ui.input_select(
                "stage",
                "AJCC stage",
                {**{code: f"Stage {label}" for code, label in _stage_lbl.items()}, **_MISSING_OPTION},
                selected="2",
            ),
            class_="patient-fields",
        ),
        ui.tags.div("Pathologic TNM", class_="input-group-title"),
        ui.tags.div(
            ui.input_select("t_stage", "Tumor (T)", {**_t_lbl, **_MISSING_OPTION}, selected="2"),
            ui.input_select("n_stage", "Nodes (N)", {**_n_lbl, **_MISSING_OPTION}, selected="0"),
            ui.input_select("m_stage", "Metastasis (M)", {**_m_lbl, **_MISSING_OPTION}, selected="0"),
            class_="patient-fields",
        ),
        ui.input_action_button("submit", "Update prediction", class_="btn-primary predict-button"),
        ui.tags.p("Unknown stages use the training-fitted imputation models.", class_="quiet-note"),
        class_="dashboard-card patient-card",
        **{"aria-label": "Patient inputs"},
    )


_prediction_tab = ui.nav_panel(
    "Prediction",
    _section_head("Patient prediction", "Compare survival estimates from both models."),
    ui.tags.div(
        _input_panel(),
        ui.tags.div(
            _card(
                "Predicted survival",
                ui.output_ui("info_bar"),
                ui.tags.div(
                    _chart("survival_curve", "desktop-chart"),
                    _chart("survival_curve_mobile", "mobile-chart"),
                    class_="prediction-figure",
                ),
                subtitle="Compare Cox PH and AFT estimates",
            ),
            _detail_panel("Estimates by time point", ui.output_ui("prob_table")),
            class_="results-stack",
        ),
        class_="prediction-workspace",
    ),
)

_cohort_tab = ui.nav_panel(
    "Cohort",
    _section_head("Cohort overview", "Pathologic stage and survival in TCGA-LUAD."),
    ui.tags.div(
        _stat_tile("Patients", f"{N_TOTAL:,}", f"{N_TRAIN} training / {N_TEST} test"),
        _stat_tile("Observed event rate", f"{EV_RATE:.1%}", "Overall survival events"),
        _stat_tile("Median follow-up", f"{MED_FU:.1f} months", "Observed"),
        class_="stat-grid",
    ),
    ui.tags.div(
        _card(
            "Cohort composition",
            _stage_table(),
            subtitle="Pathologic stage",
        ),
        _card("Candidate survival distributions", _chart("dist_plot"), subtitle="Fitted to the training cohort"),
        class_="pair-grid cohort-grid",
    ),
    ui.tags.div(
        _card(
            "Distribution fit",
            ui.tags.div(
                ui.tags.span("Selected AFT family"), ui.tags.strong(DIST["best"]),
                class_="selected-family",
            ),
            ui.tags.div(ui.output_ui("dist_table"), class_="table-frame"),
            ui.tags.p("Lower AIC indicates the preferred marginal fit among the candidates.", class_="quiet-note"),
            subtitle="Training-set AIC and BIC",
        ),
        class_="cohort-distribution",
    ),
)

_evaluation_tab = ui.nav_panel(
    "Evaluation",
    _section_head("Model evaluation", "Training performance and held-out test results."),
    ui.output_ui("perf_chips"),
    ui.tags.div(
        _card("Discrimination", _chart("perf_plot_cindex"), subtitle="C-index with 95% test bootstrap intervals"),
        _card("Discrimination over time", _chart("perf_plot_auc"), subtitle="Held-out cumulative/dynamic AUC"),
        class_="pair-grid",
    ),
    _detail_panel(
        "How to read the metrics",
        ui.tags.p(
            "Test C-index intervals use 150 bootstrap samples. Integrated Brier score uses the supported "
            "evaluation horizons at 12, 24, 36, 48, and 60 months.",
            class_="quiet-note",
        ),
    ),
)

_methods_tab = ui.nav_panel(
    "Methods",
    _section_head("Methods", "Cohort, preprocessing, models, and evaluation."),
    ui.output_ui("methods_panel"),
)

app_ui = ui.page_fluid(
    ui.tags.style(_CSS),
    ui.tags.script(_TEXT_SCALE_JS),
    ui.tags.header(
        ui.tags.div(
            ui.tags.div(
                ui.tags.span("TCGA-LUAD", class_="dataset-id"),
                ui.tags.h1("Lung cancer survival"),
                ui.tags.p(f"Overall survival prediction with Cox PH and {DIST['best']} AFT"),
            ),
            ui.tags.div(
                ui.tags.span("Research use", class_="research-label"),
                ui.tags.label("Interface text size", class_="visually-hidden", **{"for": "text-scale"}),
                ui.tags.select(
                    ui.tags.option("Standard text", value="1", selected="selected"),
                    ui.tags.option("Large text", value="1.125"),
                    ui.tags.option("Largest text", value="1.25"),
                    id="text-scale",
                    class_="text-scale-select",
                ),
                class_="masthead-tools",
            ),
            class_="page-width masthead-inner",
        ),
        class_="masthead",
    ),
    ui.tags.main(
        ui.navset_tab(_prediction_tab, _cohort_tab, _evaluation_tab, _methods_tab, id="main_tab"),
        class_="page-width main-content",
    ),
    ui.tags.footer(
        ui.tags.span("TCGA-LUAD | Overall survival"),
        ui.tags.span("For research and education only. Not for clinical decision-making."),
        class_="page-width page-footer",
    ),
    title="Lung Cancer Survival | TCGA-LUAD",
    class_="dashboard-shell",
)



# ── 6. Server ────────────────────────────────────────────────────────────────

# Function guide: server is responsible for register the reactive Shiny server logic.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def server(input, output, session):

    # ── Patient feature dataframe ─────────────────────────────────────────────

    @reactive.calc
    # Function guide: patient_feat is responsible for perform the routine-specific application step.
    # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def patient_feat() -> pd.DataFrame:
        input.submit()
        with reactive.isolate():
            try:
                row = sc.validate_clinical_inputs({
                    "age": input.age(), "stage": input.stage(), "t_stage": input.t_stage(),
                    "n_stage": input.n_stage(), "m_stage": input.m_stage(),
                })
            except ValueError as exc:
                raise SafeException(str(exc)) from None
        # Convert the validated UI values to the exact raw feature schema, then
        # apply the training-fitted imputer, spline/dummy decisions, and column
        # ordering stored in the bundle.  No patient input is persisted.
        df_pt = pd.DataFrame([row])
        # Standard deployment applies the frozen training imputers to missing
        # inputs and pools predictions across the five completed representations.
        return _IMP.transform_multiple(df_pt)

    @reactive.calc
    # Function guide: curves is responsible for perform the routine-specific application step.
    # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def curves():
        feat   = patient_feat()
        # Both models receive the same preprocessed patient row.  Their curves
        # differ only because their fitted statistical assumptions differ.
        s_cox  = _predict_curve(COX, feat)
        s_aft  = _predict_curve(AFT, feat)
        return s_cox, s_aft

    # ── Info bar ──────────────────────────────────────────────────────────────

    @render.ui
    # Function guide: info_bar is responsible for perform the routine-specific application step.
    # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def info_bar():
        cox, aft = curves()
        chips = []
        for month in (12, 36, 60):
            first = float(np.interp(month, _CURVE_T, cox))
            second = float(np.interp(month, _CURVE_T, aft))
            chips.append(
                f'<div class="milestone"><h4>{month} months</h4>'
                f'<span class="milestone-row"><span><i class="model-dot" style="background:{_COX_CLR};"></i>'
                f'Cox PH</span><strong>{first:.1%}</strong></span>'
                f'<span class="milestone-row"><span><i class="model-dot" style="background:{_AFT_CLR};"></i>'
                f'AFT</span><strong>{second:.1%}</strong></span></div>'
            )
        return ui.HTML('<div class="milestone-grid">' + "".join(chips) + "</div>")

    # ── Survival curve plot ───────────────────────────────────────────────────

    @render.ui
    # Function guide: survival_curve is responsible for prepare or evaluate survival model information.
    # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def survival_curve():
        cox, aft = curves()
        return ui.tags.img(src=charts.png(charts.survival_figure(_CURVE_T, cox, aft)),
                           alt="Patient Cox PH and Log-Logistic AFT survival projections")

    @render.ui
    # Function guide: survival_curve_mobile is responsible for prepare or evaluate survival model information.
    # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def survival_curve_mobile():
        cox, aft = curves()
        return ui.tags.img(
            src=charts.png(charts.survival_figure(_CURVE_T, cox, aft, narrow=True)),
            alt="Patient survival projections with a square mobile layout",
        )

    # ── Probability table ─────────────────────────────────────────────────────

    @render.ui
    # Function guide: prob_table is responsible for perform the routine-specific application step.
    # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def prob_table():
        s_cox, s_aft = curves()
        rows = ""
        for t in _KEY_T:
            sc = float(np.interp(t, _CURVE_T, s_cox))
            sa = float(np.interp(t, _CURVE_T, s_aft))
            diff = sa - sc
            diff_str = f'{diff * 100:+.1f}'
            rows += (
                f'<tr>'
                f'<td>{int(t):d} m</td>'
                f'<td class="val cox">{sc*100:.1f}%</td>'
                f'<td class="val aft">{sa*100:.1f}%</td>'
                f'<td style="color:{BRAND["green"] if diff>=0 else BRAND["red"]};'
                f'font-weight:600;">{diff_str}</td>'
                f'</tr>'
            )
        return ui.HTML(f"""
<div style="overflow-x:auto;margin-top:4px;">
  <table class="prob-tbl">
    <thead><tr>
      <th>Time</th>
      <th style="color:{_COX_CLR};">Cox PH</th>
      <th style="color:{_AFT_CLR};">AFT</th>
      <th>AFT minus Cox PH (pp)</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>
<p class="quiet-note">
  pp = percentage points. These estimates have not been clinically validated for individual prognosis.
</p>
""")

    # ── Distribution fitting plot (pre-rendered at import) ────────────────────

    @render.ui
    # Function guide: dist_plot is responsible for create a user-facing visualization or UI component.
    # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def dist_plot():
        return ui.tags.img(src=_DIST_SRC, alt="Training marginal survival and candidate parametric fits")

    # ── Distribution AIC table ────────────────────────────────────────────────

    @render.ui
    # Function guide: dist_table is responsible for perform the routine-specific application step.
    # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def dist_table():
        tbl = DIST["table"]
        rows = ""
        for _, row in tbl.iterrows():
            is_best = row["Distribution"] == DIST["best"]
            cls = " class='best'" if is_best else ""
            star = " ★" if is_best else ""
            rows += (
                f"<tr{cls}>"
                f"<td><strong>{row['Distribution']}{star}</strong></td>"
                f"<td class='num'>{row['AIC']:.1f}</td>"
                f"<td class='num'>{row['BIC']:.1f}</td>"
                f"<td class='num'>{row['Median (months)']:.1f}</td>"
                f"</tr>"
            )
        return ui.HTML(f"""
<div style="overflow-x:auto;margin-top:4px;">
  <table class="mtbl">
    <thead><tr>
      <th>Distribution</th><th>AIC</th><th>BIC</th><th>Median OS (m)</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>
""")

    # ── Model performance plot (pre-rendered at import) ───────────────────────

    @render.ui
    # Function guide: perf_plot_cindex is responsible for create a user-facing visualization or UI component.
    # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def perf_plot_cindex():
        return ui.tags.img(src=_CINDEX_SRC, alt="Training and test C-index with test bootstrap intervals")

    @render.ui
    # Function guide: perf_plot_auc is responsible for prepare or evaluate survival model information.
    # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def perf_plot_auc():
        return ui.tags.img(src=_AUC_SRC, alt="Held-out test time-dependent AUC for Cox PH and AFT")

    # ── Performance metric chips ──────────────────────────────────────────────

    @render.ui
    # Function guide: perf_chips is responsible for perform the routine-specific application step.
    # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def perf_chips():
        def format_score(value, digits=3):
            return f"{value:.{digits}f}" if np.isfinite(value) else "N/A"

        def score(name, value, note):
            return ui.tags.div(
                ui.tags.span(name, class_="stat-label"),
                ui.tags.strong(value, class_="stat-value"),
                ui.tags.span(note, class_="stat-detail"),
                class_="score",
            )

        def model_card(title, training, test):
            metrics = []
            for label, key, digits in (("C-index", "c_index", 3), ("IBS", "ibs", 4)):
                train_value = format_score(training.get(key, float("nan")), digits)
                test_value = format_score(test[key], digits)
                metrics.append(score(label, f"{train_value} / {test_value}", "Train / test"))
            metrics.append(score("Mean AUC", format_score(test["mean_auc"]), "Test only"))
            return _card(title, ui.tags.div(*metrics, class_="score-grid"))

        return ui.tags.div(
            model_card("Cox PH", TR_COX, RES_COX),
            model_card(f"{DIST['best']} AFT", TR_AFT, RES_AFT),
            class_="pair-grid performance-grid",
        )


    @render.ui
    # Function guide: methods_panel is responsible for perform the routine-specific application step.
    # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def methods_panel():
        def format_value(value, digits=3):
            return f"{value:.{digits}f}" if np.isfinite(value) else "N/A"

        comparison_rows = []
        for name, training, test in (
            ("Cox PH", TR_COX, RES_COX),
            (f"{DIST['best']} AFT", TR_AFT, RES_AFT),
        ):
            comparison_rows.append(ui.tags.tr(
                ui.tags.th(name, scope="row"),
                ui.tags.td(format_value(training.get("c_index", float("nan")))),
                ui.tags.td(format_value(test["c_index"])),
                ui.tags.td(f"{format_value(test['ci_lo'])}–{format_value(test['ci_hi'])}"),
                ui.tags.td(format_value(training.get("ibs", float("nan")), 4)),
                ui.tags.td(format_value(test["ibs"], 4)),
                ui.tags.td(format_value(test["mean_auc"])),
            ))
        comparison = ui.tags.table(
            ui.tags.thead(ui.tags.tr(*(
                ui.tags.th(label, scope="col") for label in (
                    "Model", "Train C-index", "Test C-index", "Test 95% CI",
                    "Train IBS", "Test IBS", "Test mean AUC",
                )
            ))),
            ui.tags.tbody(*comparison_rows),
            class_="mtbl",
        )
        feature_rows = [
            ui.tags.tr(
                ui.tags.th(FEAT_DISPLAY[column], scope="row"),
                ui.tags.td(ui.tags.code(column)),
                ui.tags.td(format_value(float(_IMP.statistics_[index]), 1)),
            )
            for index, column in enumerate(FEAT_COLS)
        ]
        features = ui.tags.table(
            ui.tags.thead(ui.tags.tr(*(
                ui.tags.th(label, scope="col") for label in (
                    "Predictor", "Model field", "Representative fill value",
                )
            ))),
            ui.tags.tbody(*feature_rows),
            class_="mtbl",
        )

        def section(title, text):
            return ui.tags.section(
                ui.tags.h4(title),
                ui.tags.p(text),
                class_="method-section",
            )

        sections = [
            section(
                "Cohort and outcome",
                f"{N_TOTAL} TCGA-LUAD patients from the NCI Genomic Data Commons. "
                f"Overall survival runs from diagnosis to death or last follow-up. "
                f"The event-stratified split contains {N_TRAIN} training and {N_TEST} test patients.",
            ),
            section(
                "Predictors and preprocessing",
                "Inputs are age, AJCC stage, and pathologic T, N, and M. "
                "Five MICE imputations are fitted on training data. Age is assessed "
                "with a restricted cubic spline, and stages use treatment coding.",
            ),
            section(
                "Cox proportional hazards",
                f"The L2 penalty ({COX._penalizer_used}) is selected by five-fold "
                "training cross-validation, with preprocessing refitted in each fold. "
                "Survival curves use a Breslow baseline hazard.",
            ),
            section(
                "Accelerated failure time",
                f"{DIST['best']} is selected by the lowest training-set AIC among "
                "Weibull, Log-Normal, Log-Logistic, and Exponential distributions.",
            ),
            section(
                "Evaluation",
                "C-index and integrated Brier score are reported for training and test sets. "
                "Test C-index intervals use 150 bootstrap samples. Test AUC uses the "
                "cumulative/dynamic definition over supported times from 12 to 60 months.",
            ),
            section(
                "Interpretation",
                "Training performance is apparent performance. The models require "
                "external validation before clinical use.",
            ),
        ]
        return ui.tags.div(
            _card(
                "Model comparison",
                ui.tags.div(comparison, class_="table-frame"),
                ui.tags.p("Training estimates are apparent performance. Test intervals use 150 bootstrap samples.",
                          class_="quiet-note"),
                subtitle=f"{N_TRAIN} training / {N_TEST_EVAL} evaluated test patients",
            ),
            ui.tags.div(
                _card("Analysis methods", ui.tags.div(*sections, class_="method-narrative")),
                _card(
                    "Predictors",
                    ui.tags.div(features, class_="table-frame"),
                    ui.tags.p(
                        "Fill values come from the first completed training imputation. "
                        "MICE predictions vary across the five chains.",
                        class_="quiet-note",
                    ),
                ),
                class_="methods-layout",
            ),
            class_="methods",
        )


app = App(app_ui, server)
