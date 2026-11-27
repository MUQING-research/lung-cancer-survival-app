# Module guide:
# - Role: Define the survival model source of truth.
# - Workflow: Centralize feature schema, preprocessing estimators, model factories, tuning, and evaluation utilities.
# - Design note: Keep deployed bundle construction and evaluation rules synchronized with the app.
"""
survival_core — NSCLC Overall Survival Prediction
==================================================
Single source of truth for survival feature schema, preprocessing estimators,
model factories, and evaluation utilities used by survival_app.py.

Clinical context
----------------
Non-Small Cell Lung Cancer (NSCLC) overall survival prediction.
N = 1 000 synthetic patients; stage distribution calibrated to SEER 2015-2019.
Survival calibrated to published stage-specific median OS (SEER):
  Stage I ≈ 84 months · Stage II ≈ 36 · Stage III ≈ 18 · Stage IV ≈ 9

Models
------
1. Cox Proportional Hazards — lifelines, L2-regularised, penaliser tuned by CV
2. Parametric AFT            — best marginal distribution by AIC
3. Random Survival Forest and DeepSurv utilities are retained for research
   experiments and are not part of the current deployed bundle.

Evaluation: bootstrap C-index (95 % CI), Integrated Brier Score,
time-dependent AUC, calibration at 12/24/36/60 months.

Simulated data only — NOT a clinical diagnostic tool.

Modeling notes
--------------
Each row represents one patient with a follow-up time and an event indicator.
The event indicator is 1 for death and 0 for right censoring.  The deployed
workflow keeps the 80/20 stratified split, imputation, functional-form checks,
and Cox penalizer selection inside the training boundary.  The Cox model and
the AFT model are evaluated on the untouched test partition with survival
metrics that respect censoring.  All prediction helpers consume the same
preprocessed feature schema used during fitting.
"""

from __future__ import annotations

# ruff: noqa: E402

import os
import platform
import tempfile
import warnings
import time
import copy
from dataclasses import dataclass
from typing import Optional

# ── ASCII temp dir (CJK Windows usernames crash joblib) ─────────────────────

# Function guide: _ensure_ascii_joblib_tmp is responsible for perform ascii joblib tmp.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _ensure_ascii_joblib_tmp() -> None:
    """Choose an ASCII-only temporary directory when the default path is unsafe.

    Some Windows joblib environments fail while serializing temporary files
    when the user profile path contains non-ASCII characters.  This startup
    guard changes only the temporary-file location; it does not alter model
    data, feature values, or statistical calculations.
    """
    if os.environ.get("JOBLIB_TEMP_FOLDER"):
        return
    try:
        tempfile.gettempdir().encode("ascii")
    except UnicodeEncodeError:
        drive = os.environ.get("SystemDrive", "C:") + os.sep
        for cand in (os.path.join(drive, "Temp"), os.path.join(drive, "joblib_tmp")):
            try:
                os.makedirs(cand, exist_ok=True)
                os.environ["JOBLIB_TEMP_FOLDER"] = cand
                return
            except OSError:
                continue

_ensure_ascii_joblib_tmp()
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # suppress OMP duplicate-lib warning

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from scipy.linalg import null_space
from scipy.interpolate import BSpline
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
import joblib

import lifelines
import autograd.numpy as anp
from lifelines.fitters import ParametricRegressionFitter
from lifelines import (
    CoxPHFitter, KaplanMeierFitter,
    WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter,
    WeibullFitter, LogNormalFitter, LogLogisticFitter, ExponentialFitter,
)

import sksurv
from sksurv.util import Surv
from sksurv.metrics import (
    concordance_index_censored,
    integrated_brier_score,
    cumulative_dynamic_auc,
)

_DEEPSURV_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torchtuples as tt
    from pycox.models import CoxPH as _PycoxCoxPH
    _DEEPSURV_AVAILABLE = True
except ImportError:
    warnings.warn("pycox/torch not available — DeepSurv will be skipped.", ImportWarning)

__version__ = "1.0.0"
SEED = 42
CELL_COLORS = [
    "#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F",
    "#8491B4", "#91D1C2", "#DC0000", "#7E6148", "#B09C85",
]


# Function guide: apply_cell_matplotlib_style is responsible for apply cell matplotlib style.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def apply_cell_matplotlib_style(**overrides) -> None:
    """Apply Cell-style matplotlib defaults for app and report figures."""
    params = {
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans", "sans-serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.grid": False,
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
        "legend.frameon": False,
    }
    params.update(overrides)
    mpl.rcParams.update(params)

# ── 1. Feature schema ────────────────────────────────────────────────────────

@dataclass(frozen=True)
# Class guide: Feature is responsible for perform the requested operation.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
class Feature:
    """Describe one clinical input and its validation/display contract.

    ``kind`` determines how the input is encoded for modeling, while ``low``,
    ``high``, and ``default`` describe the controls shown by the application.
    Keeping this metadata in one immutable record prevents the UI and the
    preprocessing code from silently drifting apart.
    """
    name: str
    display: str
    kind: str        # "cont" | "binary" | "ordinal" | "categorical"
    unit: str
    low: float
    high: float
    default: float
    description: str


FEATURES: tuple[Feature, ...] = (
    Feature("age",                  "Age",                       "cont",        "years",    30,  90,   65, "Age at diagnosis"),
    Feature("sex_male",             "Sex (Male=1)",              "binary",      "",          0,   1,    1, "Biological sex"),
    Feature("ecog_ps",              "ECOG Performance Status",   "ordinal",     "",          0,   3,    1, "0=fully active … 3=limited self-care"),
    Feature("stage",                "Stage (I–IV)",              "ordinal",     "",          1,   4,    3, "UICC 8th edition pathological stage"),
    Feature("histology",            "Histology",                 "categorical", "",          0,   2,    0, "0=Adenocarcinoma 1=Squamous 2=Other NSCLC"),
    Feature("smoking_pack_years",   "Smoking Pack-years",        "cont",        "pack-yrs",  0,  80,   20, "Cumulative tobacco exposure"),
    Feature("tumor_size_cm",        "Tumour Size",               "cont",        "cm",      0.5,  12,  3.5, "Max diameter on CT"),
    Feature("n_positive_nodes",     "Positive Node Count",       "cont",        "",          0,  20,    2, "Pathologically positive lymph nodes"),
    Feature("treatment",            "Primary Treatment",         "categorical", "",          0,   3,    1, "0=Surgery 1=Chemo 2=CRT 3=Targeted"),
    Feature("bmi",                  "BMI",                       "cont",        "kg/m²",    14,  40,   24, "Body mass index"),
    Feature("egfr_mutation",        "EGFR Mutation",             "binary",      "",          0,   1,    0, "Sensitising EGFR mutation"),
    Feature("nlr",                  "Neutrophil-Lymphocyte Ratio","cont",       "",        1.0,15.0,  3.5, "Pre-treatment systemic inflammation marker"),
)

FEATURE_NAMES = [f.name for f in FEATURES]
FEATURE_DISPLAY = {f.name: f.display for f in FEATURES}
TIME_COL  = "time_months"
EVENT_COL = "event_death"
FOLLOW_UP_MAX = 60.0  # administrative censoring at 5 years

# Weibull shape calibrated to NSCLC (mildly increasing hazard)
_WEIBULL_SHAPE = 1.30
# Stage-specific median OS (SEER 2015-2019 NSCLC, all-comers)
_STAGE_MEDIAN_OS = {1: 84.0, 2: 36.0, 3: 18.0, 4: 9.0}
_STAGE_PROBS = np.array([0.18, 0.08, 0.24, 0.50])

# Log HRs grounded in published meta-analyses
_LOG_HR = dict(
    age=0.018, sex_male=0.18, ecog_ps=0.38,
    hist_sq=0.12, hist_oth=0.20,
    smoking_pack_years=0.004,
    tumor_size_cm=0.095, n_positive_nodes=0.075,
    treat_chemo=0.35, treat_crt=0.22, treat_targeted=-0.28,
    bmi=-0.025, egfr_mutation=-0.50, nlr=0.06,
)

BRAND = dict(
    red=CELL_COLORS[0],
    blue=CELL_COLORS[1],
    green=CELL_COLORS[2],
    navy=CELL_COLORS[3],
    orange=CELL_COLORS[4],
    lavender=CELL_COLORS[5],
    mint=CELL_COLORS[6],
    crimson=CELL_COLORS[7],
    brown=CELL_COLORS[8],
    tan=CELL_COLORS[9],
    light="white",
    grid="#E2E8F0",
    edge=CELL_COLORS[5],
    ink=CELL_COLORS[3],
)

MODEL_COLORS = {
    "Cox PH":    BRAND["navy"],
    "AFT":       BRAND["green"],
    "RSF":       BRAND["orange"],
    "DeepSurv":  BRAND["red"],
}

# ── 2. Data simulation ───────────────────────────────────────────────────────

# Function guide: generate_nsclc_cohort is responsible for generate the synthetic survival cohort.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def generate_nsclc_cohort(n: int = 1000, seed: int = SEED) -> pd.DataFrame:
    """
    Simulate a realistic NSCLC cohort with clinically calibrated correlations.

    This generator is a reproducible development fixture, not a source of
    clinical evidence.  It creates covariates first, then generates event and
    censoring times from a stage-stratified Weibull proportional-hazards
    mechanism.  The returned table contains both the model predictors and the
    two survival outcome columns required by lifelines and scikit-survival.

    Biology encoded:
    - Stage IV → worse ECOG, more positive nodes, more systemic treatment
    - Squamous → heavy smoker, male, rarely EGFR-mutant
    - EGFR mutation → adenocarcinoma, female, never/light smoker, targeted therapy
    - Survival generated from Weibull PH model (stage-stratified baseline)
    """
    # A local generator makes repeated calls deterministic without modifying
    # NumPy's process-wide random state.
    rng = np.random.default_rng(seed)

    # Stage is the main baseline-risk stratifier.  The probabilities sum to
    # one and are used for every generated cohort unless a caller changes the
    # module-level calibration constants.
    stage = rng.choice([1, 2, 3, 4], size=n, p=_STAGE_PROBS)

    # Histology proportions vary by stage, so the synthetic data retain a
    # clinically plausible relationship between disease extent and subtype.
    adeno_p = np.clip(0.52 - 0.04 * (stage - 1), 0.35, 0.60)
    sq_p    = np.clip(0.29 + 0.02 * (stage - 1), 0.24, 0.38)
    oth_p   = 1 - adeno_p - sq_p
    hist_probs = np.column_stack([adeno_p, sq_p, oth_p])
    histology = np.array([rng.choice(3, p=hist_probs[i]) for i in range(n)])

    # Sex is correlated with histology through a subtype-specific probability.
    male_p = np.where(histology == 1, 0.72, 0.55)
    sex_male = rng.binomial(1, male_p)

    # Age is clipped to the supported clinical range before it enters the
    # hazard model.
    age = rng.normal(65, 10, n).clip(35, 90).round().astype(int)

    # ECOG performance status becomes progressively worse as stage increases.
    ecog_probs = np.array([
        [0.50, 0.35, 0.12, 0.03],
        [0.38, 0.40, 0.17, 0.05],
        [0.22, 0.38, 0.28, 0.12],
        [0.12, 0.30, 0.35, 0.23],
    ])
    ecog_ps = np.array([rng.choice(4, p=ecog_probs[s - 1]) for s in stage])

    # Smoking exposure is right-skewed.  A small fixed probability of zero
    # creates never-smokers while preserving heavier exposure in squamous
    # histology.
    base_py = np.where(histology == 1, 42.0, 18.0)
    smoking = rng.gamma(shape=2.0, scale=base_py / 2.0).clip(0, 80)
    smoking[rng.random(n) < 0.12] = 0.0  # 12% never-smokers

    # Tumor size is sampled from a gamma distribution whose mean increases by
    # stage, giving a positive continuous measurement.
    size_mean = np.array([2.0, 3.5, 5.0, 5.5])[(stage - 1)]
    tumor_size_cm = rng.gamma(shape=3, scale=size_mean / 3).clip(0.5, 12)

    # Positive-node counts use stage-specific Poisson means and are bounded at
    # 20 to match the input validation range.
    node_lam = {1: 0.05, 2: 1.5, 3: 5.0, 4: 7.0}
    n_positive_nodes = np.array([
        max(0, int(rng.poisson(node_lam[s]))) for s in stage
    ]).clip(0, 20)

    # Treatment is assigned from a stage-guided categorical distribution.  The
    # treatment effects are later represented by one-hot indicator columns.
    treat_probs = np.array([
        [0.80, 0.08, 0.08, 0.04],
        [0.55, 0.15, 0.20, 0.10],
        [0.10, 0.28, 0.52, 0.10],
        [0.02, 0.45, 0.25, 0.28],
    ])
    treatment = np.array([rng.choice(4, p=treat_probs[s - 1]) for s in stage])

    # BMI is approximately symmetric and is clipped to a plausible range.
    bmi = rng.normal(24.5, 4.0, n).clip(14, 42).round(1)

    # EGFR mutation is more common in adenocarcinoma, females, and light or
    # never-smokers; late-stage mutation-positive cases receive targeted
    # treatment in this synthetic mechanism.
    egfr_base = np.where(
        (histology == 0) & (sex_male == 0) & (smoking < 10), 0.42,
        np.where(histology == 0, 0.15, 0.03)
    )
    egfr_mutation = rng.binomial(1, egfr_base)
    treatment[(egfr_mutation == 1) & (stage >= 3)] = 3  # targeted for late-stage EGFR+

    # NLR increases with performance status and stage and is generated as a
    # positive gamma-distributed inflammation marker.
    nlr_mean = np.clip(2.5 + 0.35 * ecog_ps + 0.25 * (stage - 1), 1.0, 10.0)
    nlr = rng.gamma(shape=4, scale=nlr_mean / 4).clip(1.0, 15.0).round(1)

    # ── Survival generation (Weibull PH) ────────────────────────────────────
    # Baseline scale per stage → matches published median OS
    # Weibull median = scale * (ln2)^(1/shape)  →  scale = median / (ln2)^(1/shape)
    ln2_k = np.log(2) ** (1.0 / _WEIBULL_SHAPE)
    stage_scale = np.array([_STAGE_MEDIAN_OS[s] / ln2_k for s in stage])

    # The linear predictor contains log hazard-ratio contributions for all
    # non-stage covariates.  Stage is represented through the baseline scale so
    # its calibrated median survival remains interpretable.
    lp = (
        _LOG_HR["age"]               * (age - 65)
        + _LOG_HR["sex_male"]        * sex_male
        + _LOG_HR["ecog_ps"]         * ecog_ps
        + _LOG_HR["hist_sq"]         * (histology == 1)
        + _LOG_HR["hist_oth"]        * (histology == 2)
        + _LOG_HR["smoking_pack_years"] * (smoking - 20)
        + _LOG_HR["tumor_size_cm"]   * (tumor_size_cm - 3.5)
        + _LOG_HR["n_positive_nodes"]* n_positive_nodes
        + _LOG_HR["treat_chemo"]     * (treatment == 1)
        + _LOG_HR["treat_crt"]       * (treatment == 2)
        + _LOG_HR["treat_targeted"]  * (treatment == 3)
        + _LOG_HR["bmi"]             * (bmi - 24)
        + _LOG_HR["egfr_mutation"]   * egfr_mutation
        + _LOG_HR["nlr"]             * (nlr - 3.5)
    )
    # Center the linear predictor within each stage.  This makes the specified
    # stage scale correspond to the target stage-specific median instead of
    # shifting that median because of nonzero average covariate effects.
    for s in [1, 2, 3, 4]:
        mask = (stage == s)
        if mask.sum() > 0:
            lp[mask] -= lp[mask].mean()

    # Inverse-CDF sampling for a Weibull proportional-hazards model.  A larger
    # log hazard ratio shortens the generated event time.
    U = rng.uniform(0, 1, n)
    T_event = stage_scale * (-np.log(U)) ** (1.0 / _WEIBULL_SHAPE) * np.exp(-lp / _WEIBULL_SHAPE)

    # Observed follow-up is the earlier of death and censoring.  Administrative
    # censoring enforces the five-year evaluation horizon; random censoring
    # mimics loss to follow-up without revealing the event time.
    C = np.minimum(FOLLOW_UP_MAX, rng.exponential(scale=84.0, size=n))
    time_obs  = np.minimum(T_event, C).clip(0.1).round(1)
    event_obs = (T_event <= C).astype(int)

    return pd.DataFrame({
        "age":                age,
        "sex_male":           sex_male,
        "ecog_ps":            ecog_ps,
        "stage":              stage,
        "histology":          histology,
        "smoking_pack_years": smoking.round(1),
        "tumor_size_cm":      tumor_size_cm.round(1),
        "n_positive_nodes":   n_positive_nodes.astype(int),
        "treatment":          treatment,
        "bmi":                bmi,
        "egfr_mutation":      egfr_mutation,
        "nlr":                nlr,
        TIME_COL:             time_obs,
        EVENT_COL:            event_obs,
    })


# ── 3. Feature preprocessing (one-hot encode categoricals; keep ordinal/cont)

_BASE_COLS = [
    "age", "sex_male", "ecog_ps", "stage",
    "smoking_pack_years", "tumor_size_cm", "n_positive_nodes",
    "bmi", "egfr_mutation", "nlr",
]

# The deployed TCGA-LUAD model uses a smaller schema defined in
# 02_data_preprocessing/survival_preprocessing.py.  Keep the categorical
# policy explicit here so every ordered and binary clinical category is
# represented by treatment-coded dummy variables in the deployed matrix.
_DEPLOYED_CATEGORICAL_FEATURES = {"stage", "t_stage", "n_stage", "m_stage"}

# Function guide: preprocess is responsible for preprocess the requested operation.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Create the fixed model feature matrix from a clinical table.

    Continuous and binary variables are copied as numeric columns.  Histology
    and treatment are nominal categorical variables, so their non-reference
    levels are represented by explicit indicator columns.  The function does
    not estimate any distribution-dependent statistic and therefore can be
    called on train, validation, or test rows; training-only imputation and
    functional-form selection are handled by ``ClinicalPreprocessor`` when
    those operations are required.
    """
    # Keep the base order stable so a fitted model sees identical columns on
    # every split and at deployment time.
    out = df[_BASE_COLS].copy().astype(float)

    # Use the clinically meaningful reference levels: adenocarcinoma and
    # surgery are represented by all-zero indicator values.
    out["hist_squamous"] = (df["histology"] == 1).astype(float)
    out["hist_other"]    = (df["histology"] == 2).astype(float)
    out["treat_chemo"]   = (df["treatment"] == 1).astype(float)
    out["treat_crt"]     = (df["treatment"] == 2).astype(float)
    out["treat_targeted"]= (df["treatment"] == 3).astype(float)
    return out

# Function guide: PROC_COLS is responsible for perform COLS.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def PROC_COLS(df: pd.DataFrame) -> list[str]:
    return preprocess(df).columns.tolist()


# ── 4. Distribution testing (univariate; guides AFT family choice) ───────────

# Class guide: ClinicalImputer is responsible for perform the requested operation.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
class ClinicalImputer(BaseEstimator, TransformerMixin):
    """Fit training-only missing-value replacements for clinical features.

    Age is continuous and receives the training median.  Ordered and binary
    clinical codes receive the most frequent observed training category.  The
    fitted column names and statistics are stored so a test or deployment
    frame cannot silently reorder features or recompute imputation values.
    """

    # Function guide: fit is responsible for fit the requested operation.
    # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def fit(self, X, y=None):
        # Record the schema before calculating replacements.  This makes a
        # mismatch between training and prediction inputs fail explicitly.
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        values = []
        for col in X.columns:
            observed = X[col].dropna()
            if observed.empty:
                raise ValueError(f"Cannot impute entirely missing training feature: {col}")
            # Use a robust location for continuous age and the modal observed
            # category for discrete clinical codes.
            values.append(observed.median() if col == "age" else observed.mode().iloc[0])
        self.statistics_ = np.asarray(values, dtype=float)
        return self

    # Function guide: transform is responsible for transform the requested operation.
    # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def transform(self, X):
        # Apply the frozen training statistics.  Return a NumPy matrix because
        # the downstream functional-form transformer controls output columns.
        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError("Clinical feature columns differ from the fitted schema.")
        return X.astype(float).fillna(dict(zip(self.feature_names_in_, self.statistics_))).to_numpy()


# Class guide: RestrictedCubicSpline is responsible for build a natural cubic spline basis.
# Inputs: one fitted training predictor and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
class RestrictedCubicSpline(BaseEstimator, TransformerMixin):
    """Construct a restricted (natural) cubic spline basis for one variable.

    The first column is the raw linear component.  Remaining columns are
    truncated-cubic contrasts that are exactly linear outside the boundary
    knots.  Knots are estimated from the training values only and the same
    fitted basis is reused for validation, test, and deployment rows.
    """

    def __init__(self, n_knots: int = 3, quantiles=(0.05, 0.50, 0.95)):
        self.n_knots = int(n_knots)
        self.quantiles = tuple(float(value) for value in quantiles)

    def fit(self, X, y=None):
        values = np.asarray(pd.DataFrame(X).iloc[:, 0], dtype=float)
        if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
            raise ValueError("Restricted cubic spline requires finite training values.")
        if self.n_knots < 3:
            raise ValueError("Restricted cubic spline requires at least three knots.")
        if len(self.quantiles) != self.n_knots:
            raise ValueError("The number of knot quantiles must equal n_knots.")
        if any(value <= 0.0 or value >= 1.0 for value in self.quantiles):
            raise ValueError("Knot quantiles must lie strictly between zero and one.")
        knots = np.quantile(values, self.quantiles)
        if np.unique(knots).size < self.n_knots:
            unique = np.unique(values)
            if unique.size < self.n_knots:
                raise ValueError("Training values do not contain enough distinct knots.")
            positions = np.linspace(0, unique.size - 1, self.n_knots).round().astype(int)
            knots = unique[positions]
        if np.any(np.diff(knots) <= 0.0):
            raise ValueError("Restricted cubic spline knots must be strictly increasing.")
        self.knots_ = np.asarray(knots, dtype=float)
        self.lower_boundary_ = float(self.knots_[0])
        self.upper_boundary_ = float(self.knots_[-1])
        self.scale_ = max((self.upper_boundary_ - self.lower_boundary_) ** 2, 1.0)
        self.n_output_features_ = 1 + self.n_knots - 2
        self.feature_names_out_ = ["age_rcs_linear"] + [
            f"age_rcs_nonlinear_{index}"
            for index in range(1, self.n_knots - 1)
        ]
        self.n_features_in_ = 1
        return self

    def transform(self, X):
        if not hasattr(self, "knots_"):
            raise ValueError("Restricted cubic spline must be fitted before transform.")
        values = np.asarray(pd.DataFrame(X).iloc[:, 0], dtype=float)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("Restricted cubic spline requires finite values.")
        lower = self.knots_[-2]
        upper = self.knots_[-1]
        denominator = upper - lower
        basis = [values]
        for knot in self.knots_[:-2]:
            nonlinear = (
                np.maximum(values - knot, 0.0) ** 3
                - ((upper - knot) / denominator)
                * np.maximum(values - lower, 0.0) ** 3
                + ((lower - knot) / denominator)
                * np.maximum(values - upper, 0.0) ** 3
            ) / self.scale_
            basis.append(nonlinear)
        return np.column_stack(basis)

    def get_feature_names_out(self, input_features=None):
        if not hasattr(self, "feature_names_out_"):
            raise ValueError("Restricted cubic spline must be fitted before naming features.")
        return np.asarray(self.feature_names_out_, dtype=object)


# The project uses a liberal alpha for functional-form diagnostics.  The
# article's decision table is reproduced with the same two tests, while the
# threshold remains the prespecified alpha for this project.
FUNCTIONAL_FORM_ALPHA = 0.10


# Function guide: _article_style_wald_test is responsible for perform a joint Wald test.
# Inputs: a fitted Cox model and a list of coefficient names. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _article_style_wald_test(model, term_names: list[str]) -> dict:
    """Test a Cox term set in the form reported by ``rms::anova``.

    ``anova.rms`` reports a total test for all spline columns and a separate
    test for the nonlinear columns.  This helper applies the same joint-Wald
    construction to an unpenalized ``lifelines`` Cox fit, without retaining
    the fitted model (and therefore without placing patient rows in the
    deployment bundle).
    """
    available = [name for name in term_names if name in model.params_.index]
    if not available:
        raise ValueError("No requested terms are present in the Cox fit.")
    beta = model.params_.loc[available].to_numpy(dtype=float)
    covariance = model.variance_matrix_.loc[available, available].to_numpy(dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    statistic = float(beta @ np.linalg.pinv(covariance) @ beta)
    statistic = max(0.0, statistic)
    degrees = int(len(available))
    return {
        "chi2": statistic,
        "df": degrees,
        "p": float(stats.chi2.sf(statistic, degrees)),
        "terms": available,
    }


# Function guide: _article_style_covariates is responsible for build article-style adjustment covariates.
# Inputs: an imputed feature frame and fitted category levels. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _article_style_covariates(values: pd.DataFrame, categories: dict[str, list[int]]) -> pd.DataFrame:
    """Build the covariate block used alongside the age RCS term.

    The article fits the target variable as an RCS while retaining the same
    covariates in every model.  Here categorical clinical predictors are
    represented by treatment dummies, exactly as in the deployed design.
    """
    result = pd.DataFrame(index=values.index)
    for name in values.columns:
        if name == "age":
            continue
        if name in _DEPLOYED_CATEGORICAL_FEATURES:
            levels = categories.get(name, sorted(values[name].astype(int).unique().tolist()))
            for level in levels[1:]:
                result[f"{name}_{int(level)}"] = (values[name] == level).astype(float)
        else:
            result[name] = values[name].astype(float)
    return result


# Function guide: _article_style_form_decision is responsible for classify the two article-style P values.
# Inputs: overall and nonlinear P values plus the diagnostic threshold. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _article_style_form_decision(
    p_total: float,
    p_nonlinear: float,
    alpha: float = FUNCTIONAL_FORM_ALPHA,
) -> tuple[str, str]:
    """Return the article's association label and selected age representation."""
    total_significant = bool(p_total < alpha)
    nonlinear_significant = bool(p_nonlinear < alpha)
    if total_significant and nonlinear_significant:
        return "nonlinear_association", "rcs"
    if total_significant and not nonlinear_significant:
        return "linear_association", "linear"
    if not total_significant and not nonlinear_significant:
        return "no_evidence_of_association", "linear"
    # The article does not specify this uncommon combination.  Retaining the
    # nonlinear term avoids forcing a linear form when the nonlinear component
    # itself is statistically detectable; the exceptional state is recorded.
    return "nonlinear_shape_without_overall_significance", "rcs"


# Class guide: ClinicalPreprocessor is responsible for perform the requested operation.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
class ClinicalPreprocessor(BaseEstimator, TransformerMixin):
    """Fit imputation and functional-form decisions on one training fold.

    The preprocessor first imputes raw clinical variables, then fits the
    article-style three-knot restricted cubic spline diagnostic for age.
    ``P`` tests the total age effect and ``P nonlinear`` tests the nonlinear
    spline columns jointly.  The selected representation is then frozen for
    the final design matrix.  All deployed
    categorical variables, including binary M stage, use treatment-coded
    dummies.  Categorical variables therefore receive an encoding audit, not
    a continuous-linearity test: each level has its own coefficient relative
    to the reference level.  All learned decisions and spline knots are
    retained so the same transformation can be applied to validation, test,
    and deployment rows without leakage.
    """

    # Function guide: fit is responsible for fit the requested operation.
    # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def fit(self, X, y):
        # Fit the imputer only on the current training fold.  The survival
        # outcome is used below for functional-form diagnostics, not for
        # calculating missing-value replacements.
        self.imputer_ = ClinicalImputer().fit(X)
        self.feature_names_in_ = self.imputer_.feature_names_in_
        self.statistics_ = self.imputer_.statistics_
        values = pd.DataFrame(self.imputer_.transform(X), columns=X.columns, index=X.index)
        self.forms_, self.categories_, self.form_checks_, self.encoding_checks_ = {}, {}, [], []
        self.age_rcs_ = None
        self.age_spline_ = None
        self.age_rcs_diagnostic_ = None

        # Establish all categorical encodings before fitting the age model so
        # the total and nonlinear tests adjust for the same covariates as the
        # final Cox model, just as the article's cph() formula does.
        for name in X.columns:
            if name == "age":
                continue
            if name in _DEPLOYED_CATEGORICAL_FEATURES:
                # All categorical variables are now treatment-coded dummies,
                # including binary M stage.  The first observed training
                # level is the reference; no ordinal integer coding is used
                # in the final design matrix.
                self.forms_[name] = "dummy"
                self.categories_[name] = sorted(values[name].astype(int).unique().tolist())
                self.encoding_checks_.append({
                    "variable": name,
                    "method": "Treatment-coded one-hot encoding",
                    "reference_level": self.categories_[name][0] if self.categories_[name] else None,
                    "levels": self.categories_[name],
                    "linearity_test": "not applicable to dummy-coded categorical variables",
                })
                continue

            # Preserve a safe linear fallback for any non-deployed feature
            # used by synthetic research fixtures.
            self.forms_[name] = "linear"

        self.forms_["age"] = "linear"
        try:
            age_rcs = RestrictedCubicSpline(n_knots=3).fit(values[["age"]])
            basis = pd.DataFrame(
                age_rcs.transform(values[["age"]]),
                index=values.index,
                columns=age_rcs.get_feature_names_out().tolist(),
            )
            covariates = _article_style_covariates(values, self.categories_)
            outcome = pd.DataFrame(y)[[TIME_COL, EVENT_COL]].copy()
            outcome.index = values.index
            model_frame = pd.concat([basis, covariates, outcome], axis=1)
            article_fit = CoxPHFitter().fit(model_frame, TIME_COL, EVENT_COL)
            spline_names = age_rcs.get_feature_names_out().tolist()
            total_test = _article_style_wald_test(article_fit, spline_names)
            nonlinear_test = _article_style_wald_test(article_fit, spline_names[1:])
            association, selected_form = _article_style_form_decision(
                total_test["p"], nonlinear_test["p"], FUNCTIONAL_FORM_ALPHA,
            )
            self.forms_["age"] = selected_form
            self.age_rcs_diagnostic_ = age_rcs
            if selected_form == "rcs":
                self.age_rcs_ = age_rcs
                self.age_spline_ = age_rcs
            self.form_checks_.append({
                "variable": "age",
                "method": "Article-style Cox RCS overall and nonlinear Wald tests",
                "rcs_knots": age_rcs.knots_.tolist(),
                "p_total": total_test["p"],
                "chi2_total": total_test["chi2"],
                "df_total": total_test["df"],
                "p_nonlinear": nonlinear_test["p"],
                "chi2_nonlinear": nonlinear_test["chi2"],
                "df_nonlinear": nonlinear_test["df"],
                # Keep the short names used by earlier audit consumers while
                # exposing the two article-style P values explicitly.
                "p": total_test["p"],
                "diagnostic_decision": association,
                "decision": selected_form,
                "deployed_form": "restricted_cubic_spline" if selected_form == "rcs" else "linear",
                "alpha": FUNCTIONAL_FORM_ALPHA,
                "plot_reference": "Training median age; HR=1",
            })
        except (KeyError, ValueError, lifelines.exceptions.ConvergenceError, np.linalg.LinAlgError) as exc:
            # If the full adjusted RCS fit cannot be estimated, keep the
            # deployable representation linear and record the failed audit.
            self.forms_["age"] = "linear"
            self.form_checks_.append({
                "variable": "age",
                "method": "Article-style Cox RCS overall and nonlinear Wald tests",
                "error": str(exc),
                "fallback": "linear",
                "decision": "linear",
                "deployed_form": "linear",
                "alpha": FUNCTIONAL_FORM_ALPHA,
            })
        self.output_columns_ = self.transform(X).columns.tolist()
        return self

    # Function guide: transform is responsible for transform the requested operation.
    # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def transform(self, X):
        # Reuse the imputer and the representation chosen during fit.  The
        # result is rebuilt in the original feature order on every call.
        values = pd.DataFrame(self.imputer_.transform(X), columns=X.columns, index=X.index)
        result = pd.DataFrame(index=X.index)
        for name in X.columns:
            form = self.forms_[name]
            if form == "rcs":
                # Apply the training-fitted restricted spline to new patients.
                basis = self.age_rcs_.transform(values[[name]])
                for j in range(basis.shape[1]):
                    result[self.age_rcs_.feature_names_out_[j]] = basis[:, j]
            elif form == "dummy":
                # Map unseen or unsupported levels to the training imputation
                # value before creating non-reference dummy columns.
                levels = self.categories_[name]
                replacement = self.statistics_[list(self.feature_names_in_).index(name)]
                coded = values[name].where(values[name].isin(levels), replacement)
                for level in levels[1:]:
                    result[f"{name}_{int(level)}"] = (coded == level).astype(float)
            else:
                result[name] = values[name]
        return result


# Class guide: MultipleClinicalPreprocessor is responsible for training-only multiple imputation.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
class MultipleClinicalPreprocessor(BaseEstimator, TransformerMixin):
    """Fit a small MICE-style ensemble of training-only imputers.

    ``IterativeImputer`` is used with posterior sampling to create several
    plausible completed baseline-feature datasets.  Continuous age is kept
    continuous; ordered and binary clinical codes are rounded and clipped to
    their documented domains after each chained-equation draw.  A separate
    ``ClinicalPreprocessor`` is fitted to each completed dataset so functional
    form decisions remain tied to the corresponding imputation.

    The imputation model intentionally uses baseline predictors only.  This is
    suitable for deployment because a future outcome is unavailable when an
    individual patient is scored.  The fitted objects store model parameters,
    not patient rows.
    """

    _CATEGORICAL_BOUNDS = {
        "stage": (1, 4),
        "t_stage": (0, 4),
        "n_stage": (0, 3),
        "m_stage": (0, 1),
    }

    def __init__(self, n_imputations: int = 5, random_state: int = SEED):
        self.n_imputations = int(n_imputations)
        self.random_state = int(random_state)

    def _postprocess(self, values: np.ndarray) -> np.ndarray:
        """Round categorical draws while preserving the continuous age scale."""
        result = np.asarray(values, dtype=float).copy()
        for index, name in enumerate(self.feature_names_in_):
            if name in self._CATEGORICAL_BOUNDS:
                lo, hi = self._CATEGORICAL_BOUNDS[name]
                result[:, index] = np.clip(np.rint(result[:, index]), lo, hi)
            elif name == "age":
                result[:, index] = np.clip(result[:, index], 18.0, 100.0)
        return result

    def fit(self, X, y=None):
        """Fit independent posterior-sampling imputers on the training rows."""
        if self.n_imputations < 2:
            raise ValueError("Multiple imputation requires at least two imputations.")
        frame = pd.DataFrame(X).copy()
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        bounds = [self._CATEGORICAL_BOUNDS.get(name, (18, 100)) for name in self.feature_names_in_]
        self.imputers_ = []
        self.preprocessors_ = []
        self.imputation_seeds_ = []
        for index in range(self.n_imputations):
            seed = self.random_state + index
            imputer = IterativeImputer(
                estimator=BayesianRidge(),
                max_iter=20,
                sample_posterior=True,
                random_state=seed,
                initial_strategy="median",
                min_value=np.array([bound[0] for bound in bounds], dtype=float),
                max_value=np.array([bound[1] for bound in bounds], dtype=float),
                skip_complete=True,
            )
            imputer.fit(frame)
            completed = self._postprocess(imputer.transform(frame))
            completed_frame = pd.DataFrame(
                completed, columns=frame.columns, index=frame.index,
            )
            preprocessor = ClinicalPreprocessor().fit(
                completed_frame,
                y if y is not None else pd.DataFrame(index=frame.index),
            ) if y is not None else ClinicalPreprocessor().fit(
                completed_frame,
                pd.DataFrame({TIME_COL: np.ones(len(frame)), EVENT_COL: np.ones(len(frame))}, index=frame.index),
            )
            self.imputers_.append(imputer)
            self.preprocessors_.append(preprocessor)
            self.imputation_seeds_.append(seed)

        # Use one common age representation across the imputed datasets so the
        # ensemble has a stable design schema.  Each imputation first follows
        # the article's overall/nonlinear P-value rule; the common deployed
        # form is then the majority decision across the training imputations.
        age_checks = []
        for index, preprocessor in enumerate(self.preprocessors_):
            for check in preprocessor.form_checks_:
                if check.get("variable") == "age":
                    item = dict(check)
                    item["imputation"] = index + 1
                    item["seed"] = self.imputation_seeds_[index]
                    age_checks.append(item)
        valid_checks = [
            item for item in age_checks
            if isinstance(item.get("p_total"), (int, float))
            and isinstance(item.get("p_nonlinear"), (int, float))
            and np.isfinite(item["p_total"])
            and np.isfinite(item["p_nonlinear"])
        ]
        valid_p_total = [float(item["p_total"]) for item in valid_checks]
        valid_p_nonlinear = [float(item["p_nonlinear"]) for item in valid_checks]
        rcs_votes = sum(item.get("decision") == "rcs" for item in valid_checks)
        vote_threshold = max(1, int(np.ceil(len(valid_checks) / 2.0)))
        common_form = "rcs" if rcs_votes >= vote_threshold else "linear"
        self.functional_form_summary_ = {
            "variable": "age",
            "method": "Multiple-imputation article-style Cox RCS Wald tests",
            "alpha": FUNCTIONAL_FORM_ALPHA,
            "n_imputations": len(valid_checks),
            "p_total_values": valid_p_total,
            "p_nonlinear_values": valid_p_nonlinear,
            "p_values": valid_p_total,
            "n_rcs_preferred": int(rcs_votes),
            "vote_threshold": int(vote_threshold),
            "decision_rule": (
                "Within each imputation, P_total < alpha and P_nonlinear < alpha "
                "select RCS; otherwise select linear. Use the majority form "
                "across valid training imputations for the deployed ensemble."
            ),
            "decision": common_form,
            "deployed_form": "restricted_cubic_spline" if common_form == "rcs" else "linear",
            "median_p_total": float(np.median(valid_p_total)) if valid_p_total else None,
            "median_p_nonlinear": float(np.median(valid_p_nonlinear)) if valid_p_nonlinear else None,
            "min_p_total": float(np.min(valid_p_total)) if valid_p_total else None,
            "max_p_total": float(np.max(valid_p_total)) if valid_p_total else None,
            "min_p_nonlinear": float(np.min(valid_p_nonlinear)) if valid_p_nonlinear else None,
            "max_p_nonlinear": float(np.max(valid_p_nonlinear)) if valid_p_nonlinear else None,
        }
        for index, preprocessor in enumerate(self.preprocessors_):
            completed = self._postprocess(
                copy.deepcopy(self.imputers_[index]).transform(frame)
            )
            completed_frame = pd.DataFrame(
                completed, columns=frame.columns, index=frame.index,
            )
            diagnostic_rcs = RestrictedCubicSpline(n_knots=3).fit(
                completed_frame[["age"]]
            )
            preprocessor.forms_["age"] = common_form
            preprocessor.age_rcs_diagnostic_ = diagnostic_rcs
            preprocessor.age_rcs_ = diagnostic_rcs if common_form == "rcs" else None
            preprocessor.age_spline_ = preprocessor.age_rcs_
            preprocessor.output_columns_ = preprocessor.transform(completed_frame).columns.tolist()

        # These aliases keep existing audit and display code compatible while
        # making the first imputation explicit as a representative schema.
        representative = self.preprocessors_[0]
        self.imputer_ = representative.imputer_
        self.statistics_ = representative.statistics_
        self.forms_ = representative.forms_
        self.categories_ = representative.categories_
        # Keep both the pooled decision and every imputation-specific result
        # in the audit log.  The per-imputation records make it possible to
        # review how sensitive the functional-form decision is to imputation.
        self.form_checks_ = [self.functional_form_summary_] + age_checks
        self.encoding_checks_ = representative.encoding_checks_
        self.age_rcs_ = getattr(representative, "age_rcs_", None)
        self.age_spline_ = getattr(representative, "age_spline_", None)
        self.output_columns_ = representative.output_columns_
        self.missing_value_rule_ = "MICE posterior sampling with five imputations"
        return self

    def _completed_frames(self, X) -> list[pd.DataFrame]:
        frame = pd.DataFrame(X).copy()
        if list(frame.columns) != list(self.feature_names_in_):
            raise ValueError("Clinical feature columns differ from the fitted schema.")
        completed_frames = []
        for imputer in self.imputers_:
            # Posterior sampling updates the estimator's RNG state during
            # transform.  Use a copy so repeated predictions remain
            # deterministic and the deployment bundle is not mutated.
            completed = self._postprocess(copy.deepcopy(imputer).transform(frame))
            completed_frames.append(pd.DataFrame(
                completed, columns=frame.columns, index=frame.index,
            ))
        return completed_frames

    def transform_multiple(self, X) -> list[pd.DataFrame]:
        """Return one model-ready feature frame per imputed dataset."""
        return [
            preprocessor.transform(completed)
            for preprocessor, completed in zip(
                self.preprocessors_, self._completed_frames(X)
            )
        ]

    def transform_complete_multiple(self, X) -> list[pd.DataFrame]:
        """Transform complete rows without invoking an imputer."""
        frame = pd.DataFrame(X).copy()
        if frame.isna().any().any():
            raise ValueError("Complete-case transformation received missing predictors.")
        if list(frame.columns) != list(self.feature_names_in_):
            raise ValueError("Clinical feature columns differ from the fitted schema.")
        return [
            preprocessor.transform(frame)
            for preprocessor in self.preprocessors_
        ]

    def transform(self, X):
        """Return the representative first-imputation frame for compatibility."""
        return self.transform_multiple(X)[0]


# Class guide: SurvivalEnsemble is responsible for pool predictions across imputations.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
class SurvivalEnsemble:
    """Pool survival predictions from models fitted to multiple imputations."""

    def __init__(self, models: list, model_type: str):
        if not models:
            raise ValueError("At least one fitted model is required.")
        self.models = list(models)
        self.model_type = model_type

    @property
    def _penalizer_used(self):
        return getattr(self.models[0], "_penalizer_used", np.nan)

    @property
    def _cv_diagnostics(self):
        return getattr(self.models[0], "_cv_diagnostics", [])

    @property
    def concordance_index_(self):
        return float(np.mean([
            getattr(model, "concordance_index_", np.nan)
            for model in self.models
        ]))

    def _pairs(self, features) -> list[tuple[object, pd.DataFrame]]:
        if isinstance(features, (list, tuple)):
            if len(features) != len(self.models):
                raise ValueError("One feature frame is required per imputed model.")
            return list(zip(self.models, features))
        # A single frame is a compatibility path for tests and known complete
        # profiles.  It is evaluated by the representative first model; true
        # pooled prediction passes one frame per imputation.
        return [(self.models[0], features)]

    def predict_partial_hazard(self, features):
        values = [
            np.asarray(model.predict_partial_hazard(frame), dtype=float).reshape(-1)
            for model, frame in self._pairs(features)
        ]
        return pd.Series(np.mean(values, axis=0))

    def predict_median(self, features):
        values = [
            np.asarray(model.predict_median(frame), dtype=float).reshape(-1)
            for model, frame in self._pairs(features)
        ]
        return pd.Series(np.mean(values, axis=0))

    def predict_expectation(self, features):
        values = [
            np.asarray(model.predict_expectation(frame), dtype=float).reshape(-1)
            for model, frame in self._pairs(features)
        ]
        return pd.Series(np.mean(values, axis=0))

    def predict_survival_function(self, features, times=None):
        pairs = self._pairs(features)
        models = [pair[0] for pair in pairs]
        frames = [pair[1] for pair in pairs]
        if times is None:
            times = np.asarray(
                self.models[0].predict_survival_function(frames[0]).index,
                dtype=float,
            )
        times = np.asarray(times, dtype=float)
        matrices = [
            _surv_matrix_lifelines(model, frame, times)
            for model, frame in zip(models, frames)
        ]
        pooled = np.mean(matrices, axis=0)
        return pd.DataFrame(
            pooled.T,
            index=times,
            columns=range(pooled.shape[0]),
        )

    def survival_function_at_times(self, features, times):
        sf = self.predict_survival_function(features, times=times)
        return sf.iloc[:, 0].to_numpy(dtype=float)


# Function guide: validate_clinical_inputs is responsible for validate clinical inputs.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def validate_clinical_inputs(values):
    """Validate server-side values before invoking either prediction model.

    This function checks only fields shared by the deployed clinical form.
    It converts observed values to floats, preserves recognized missing tokens
    as NaN for the training-fitted imputer, and enforces the allowed age and
    stage-code domains before prediction.  Model-specific preprocessing
    remains in the fitted model bundle.
    """
    result = {}
    levels = {"stage": {1, 2, 3, 4}, "t_stage": {0, 1, 2, 3, 4}, "n_stage": {0, 1, 2, 3}, "m_stage": {0, 1}}
    for key in ("age", "stage", "t_stage", "n_stage", "m_stage"):
        raw = values.get(key)
        missing_token = raw is None or (
            isinstance(raw, str)
            and raw.strip().lower() in {"", "missing", "unknown", "na", "n/a", "nan", "null"}
        )
        if missing_token:
            result[key] = np.nan
            continue
        # Convert once so UI strings and numeric inputs follow the same path.
        try:
            result[key] = float(raw)
        except (ValueError, TypeError, KeyError):
            raise ValueError(f"Enter a valid value for {key.replace('_', ' ')}.") from None
        if not np.isfinite(result[key]):
            if np.isnan(result[key]):
                result[key] = np.nan
                continue
            raise ValueError(f"Enter a finite value for {key.replace('_', ' ')}.")
        # Apply domain checks after conversion.  A value that is numeric but
        # outside a clinically supported category is still invalid input.
        if key == "age" and not 18 <= result[key] <= 100:
            raise ValueError("Age must be between 18 and 100 years.")
        if key in levels and result[key] not in levels[key]:
            raise ValueError(f"Select a valid {key.replace('_', ' ')} category.")
    return result


# Function guide: audit_clinical_preprocessing is responsible for audit clinical preprocessing.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def audit_clinical_preprocessing(df_all, df_train, features, cox, imputer):
    """Record training-only diagnostics for an existing survival specification.

    Logistic GAM tests do not accommodate censored survival outcomes. These
    checks instead use Cox partial likelihood and Schoenfeld residual tests.
    They document limitations of the prespecified deployable model and do not
    tune the model against the held-out cohort.
    """
    from lifelines.statistics import proportional_hazard_test
    decisions = {}
    raw_names = list(imputer.feature_names_in_)
    for i, name in enumerate(raw_names):
        train = df_train[name]
        kind = "continuous" if name == "age" else ("binary" if name == "m_stage" else "ordinal")
        decisions[name] = {
            "type": kind, "unit": "years" if name == "age" else "category code",
            "missing_count_train": int(train.isna().sum()), "missing_rate_train": float(train.isna().mean()),
            "missing_rate_by_event_train": {str(k): float(v) for k, v in df_train.groupby(EVENT_COL)[name].apply(lambda x: x.isna().mean()).items()},
            "missing_value_rule": "MICE posterior sampling across five imputations",
            "imputation_value": float(imputer.statistics_[i]),
            "transformation": (
                "restricted_cubic_spline" if name == "age" and imputer.forms_[name] == "rcs"
                else "linear" if name == "age" else "raw"
            ),
            "functional_form": imputer.forms_[name],
            "encoding": (
                "dummy, drop first" if imputer.forms_[name] == "dummy"
                else "restricted cubic spline basis" if name == "age" and imputer.forms_[name] == "rcs"
                else "linear age" if name == "age" else "integer code"
            ),
            "reference_level": imputer.categories_.get(name, [None])[0],
            "training_levels": imputer.categories_.get(name),
            "unseen_category": "Map to training mode before encoding",
            "centring_constant": None,
            "cut_points": imputer.age_rcs_.knots_.tolist() if name == "age" and imputer.forms_[name] == "rcs" else None,
            "sparse_levels_train": {str(k): int(v) for k, v in train.value_counts().items() if v < 10} if name != "age" else {},
            "sparse_merge_decision": "No merge: retain clinically distinct ordered stage categories.",
            "missing_indicator": "Used for missingness audit only; not included as a prediction feature.",
        }
    age = df_train["age"].dropna().to_numpy()
    diagnostics = {
        "scope": "Training data only; survival-specific diagnostics, not logistic GAM diagnostics.",
        "model_specification": "Age functional form follows the article-style Cox RCS diagnostic: overall P and P nonlinear are joint Wald tests from an adjusted three-knot RCS fit; the common selected form is applied across training imputations and refitted within CV folds.",
        "age_distribution": {"shapiro_p": float(stats.shapiro(age).pvalue), "skewness": float(stats.skew(age)), "kurtosis": float(stats.kurtosis(age))},
        "missing_per_row_train": {str(k): int(v) for k, v in df_train[raw_names].isna().sum(axis=1).value_counts().items()},
        "identical_feature_outcome_rows": int(df_all.drop(columns=["case_id"], errors="ignore").duplicated().sum()),
        "duplicates_policy": "Deduplicate case identifiers; retain different cases sharing values.",
        "cox_cv": cox._cv_diagnostics,
        "functional_form_checks": imputer.form_checks_,
        "categorical_encoding_checks": imputer.encoding_checks_,
        "limitations": [
            "Functional-form screening is exploratory; small categories can make likelihood-ratio tests unstable.",
            "Overall AJCC stage overlaps biologically with T, N and M; regularization does not remove this redundancy.",
            "MICE is the primary missing-predictor strategy; complete-case and MNAR sensitivity analyses remain important because M stage is frequently missing.",
            "Internal random split evaluation does not establish external clinical validity.",
            "Median observed follow-up is a descriptive observed-time median, not reverse Kaplan-Meier follow-up.",
        ],
    }
    fit_frame = features.copy()
    fit_frame[TIME_COL] = df_train[TIME_COL].to_numpy()
    fit_frame[EVENT_COL] = df_train[EVENT_COL].to_numpy()
    try:
        ph = proportional_hazard_test(cox, fit_frame, time_transform="rank").summary
        diagnostics["proportional_hazards"] = {str(k): float(v) for k, v in ph["p"].items()}
    except (ValueError, np.linalg.LinAlgError) as exc:
        diagnostics["proportional_hazards_error"] = str(exc)
    imputed_raw = pd.DataFrame(imputer.imputer_.transform(df_train[raw_names]), columns=raw_names)
    diagnostics["correlations_spearman"] = imputed_raw.corr(method="spearman").to_dict()
    return decisions, diagnostics


# Function guide: _cox_breslow_terms is responsible for one Cox partial-likelihood evaluation.
# Inputs: a design matrix, follow-up times, event indicators, and coefficients. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _prepare_cox_breslow(design, durations, events):
    """Precompute sorted risk-set indices for repeated Cox evaluations."""
    design = np.asarray(design, dtype=float)
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=bool)
    order = np.argsort(durations, kind="mergesort")
    sorted_durations = durations[order]
    sorted_design = design[order]
    sorted_events = events[order]
    event_times = np.unique(sorted_durations[sorted_events])
    event_starts = np.searchsorted(sorted_durations, event_times, side="left")
    event_indices, event_counts, event_design_sums = [], [], []
    for event_time in event_times:
        event_mask = sorted_events & (sorted_durations == event_time)
        event_indices.append(np.flatnonzero(event_mask))
        event_counts.append(int(event_mask.sum()))
        event_design_sums.append(sorted_design[event_mask].sum(axis=0))
    return {
        "sorted_design": sorted_design,
        "sorted_durations": sorted_durations,
        "event_starts": event_starts,
        "event_indices": event_indices,
        "event_counts": np.asarray(event_counts, dtype=float),
        "event_design_sums": np.asarray(event_design_sums, dtype=float),
    }


# Function guide: _cox_breslow_terms is responsible for one Cox partial-likelihood evaluation.
# Inputs: a design matrix, follow-up times, event indicators, and coefficients. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _cox_breslow_terms(design, durations, events, coefficients, prepared=None):
    """Return Breslow partial log-likelihood, score, and observed information."""
    if prepared is None:
        prepared = _prepare_cox_breslow(design, durations, events)
    sorted_design = prepared["sorted_design"]
    coefficients = np.asarray(coefficients, dtype=float)
    eta = sorted_design @ coefficients
    eta_shift = float(np.max(eta))
    exp_eta = np.exp(eta - eta_shift)
    cumulative_exp = np.cumsum(exp_eta[::-1])[::-1]
    cumulative_x = np.cumsum(
        (exp_eta[:, None] * sorted_design)[::-1], axis=0,
    )[::-1]
    weighted_design = (
        exp_eta[:, None, None]
        * sorted_design[:, :, None]
        * sorted_design[:, None, :]
    )
    cumulative_xx = np.cumsum(weighted_design[::-1], axis=0)[::-1]
    log_likelihood = 0.0
    score = np.zeros(sorted_design.shape[1], dtype=float)
    information = np.zeros((sorted_design.shape[1], sorted_design.shape[1]), dtype=float)
    for start, indices, event_count, event_sum in zip(
        prepared["event_starts"],
        prepared["event_indices"],
        prepared["event_counts"],
        prepared["event_design_sums"],
    ):
        denominator = cumulative_exp[start]
        log_denominator = float(np.log(denominator) + eta_shift)
        weighted_mean = cumulative_x[start] / denominator
        weighted_second = cumulative_xx[start] / denominator
        log_likelihood += float(eta[indices].sum()) - event_count * log_denominator
        score += event_sum - event_count * weighted_mean
        information += event_count * (
            weighted_second - np.outer(weighted_mean, weighted_mean)
        )
    information = (information + information.T) / 2.0
    return log_likelihood, score, information


# Function guide: _fit_cox_penalized is responsible for fit a penalized Cox partial-likelihood model.
# Inputs: a design matrix, survival outcome, and quadratic penalty. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _fit_cox_penalized(design, durations, events, penalty=None, lam=0.0, initial=None):
    """Fit a Cox model with an optional quadratic smoothness penalty."""
    design = np.asarray(design, dtype=float)
    if penalty is None:
        penalty = np.zeros((design.shape[1], design.shape[1]), dtype=float)
    penalty = np.asarray(penalty, dtype=float)
    prepared = _prepare_cox_breslow(design, durations, events)
    coefficients = (
        np.zeros(design.shape[1], dtype=float)
        if initial is None else np.asarray(initial, dtype=float).copy()
    )
    converged = False
    message = "Newton iteration limit reached."
    for _ in range(100):
        log_likelihood, score, information = _cox_breslow_terms(
            design, durations, events, coefficients, prepared=prepared,
        )
        penalized_objective = float(
            log_likelihood - 0.5 * lam * coefficients @ penalty @ coefficients
        )
        gradient = score - lam * penalty @ coefficients
        hessian = information + lam * penalty + np.eye(design.shape[1]) * 1e-9
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        if not np.all(np.isfinite(step)):
            raise ValueError("Cox partial-likelihood optimization returned non-finite values.")
        if np.linalg.norm(gradient, ord=np.inf) < 1e-7:
            converged = True
            message = "Newton iteration converged by gradient tolerance."
            break
        step_scale = 1.0
        improved = False
        for _ in range(30):
            candidate = coefficients + step_scale * step
            candidate_ll, _, _ = _cox_breslow_terms(
                design, durations, events, candidate, prepared=prepared,
            )
            candidate_objective = float(
                candidate_ll - 0.5 * lam * candidate @ penalty @ candidate
            )
            if np.isfinite(candidate_objective) and candidate_objective >= penalized_objective:
                coefficients = candidate
                improved = True
                break
            step_scale *= 0.5
        if not improved or np.linalg.norm(step_scale * step) < 1e-8:
            converged = True
            message = "Newton iteration converged by step tolerance."
            break
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("Cox partial-likelihood optimization returned non-finite values.")
    log_likelihood, score, information = _cox_breslow_terms(
        design, durations, events, coefficients, prepared=prepared,
    )
    return {
        "coefficients": coefficients,
        "log_likelihood": float(log_likelihood),
        "score": score,
        "information": information,
        "converged": converged,
        "message": message,
    }


# Function guide: _make_cox_gam_basis is responsible for build a penalized smooth basis for Cox-GAM exploration.
# Inputs: one continuous predictor. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _make_cox_gam_basis(values, n_splines=10, degree=3):
    """Build a centered P-spline basis without the fixed three-knot transformer."""
    values = np.asarray(values, dtype=float)
    lower, upper = float(np.min(values)), float(np.max(values))
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError("A Cox-GAM smooth requires a varying finite predictor.")
    n_splines = int(min(n_splines, max(4, np.unique(values).size - 1)))
    n_interior = n_splines - degree - 1
    interior = np.linspace(lower, upper, n_interior + 2)[1:-1]
    knots = np.concatenate([
        np.repeat(lower, degree + 1),
        interior,
        np.repeat(upper, degree + 1),
    ])
    raw_basis = BSpline.design_matrix(
        values, knots, degree, extrapolate=True,
    ).toarray()
    # Remove the constant direction absorbed by the Cox baseline hazard.
    contrast = null_space(np.ones((1, raw_basis.shape[1])))
    basis = raw_basis @ contrast
    difference = np.diff(np.eye(raw_basis.shape[1]), n=2, axis=0)
    penalty = contrast.T @ difference.T @ difference @ contrast

    def transform(new_values):
        new_values = np.asarray(new_values, dtype=float)
        return BSpline.design_matrix(
            new_values, knots, degree, extrapolate=True,
        ).toarray() @ contrast

    return basis, transform, penalty


# Function guide: _fit_cox_gam_1d is responsible for fit one-dimensional Cox-GAM smooths.
# Inputs: one predictor and a censored survival outcome. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _fit_cox_gam_1d(predictor, durations, events, n_splines=10, lambda_grid=None):
    """Fit a penalized one-dimensional Cox-GAM smooth by partial-likelihood AIC."""
    predictor = np.asarray(predictor, dtype=float)
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=bool)
    if lambda_grid is None:
        lambda_grid = np.logspace(-3, 3, 7)
    basis, transform, penalty = _make_cox_gam_basis(predictor, n_splines=n_splines)
    candidates = []
    initial = None
    for lam in np.asarray(lambda_grid, dtype=float):
        fitted = _fit_cox_penalized(
            basis,
            durations,
            events,
            penalty=penalty,
            lam=float(lam),
            initial=initial,
        )
        initial = fitted["coefficients"]
        information = fitted["information"]
        regularized_information = information + float(lam) * penalty
        regularized_information += np.eye(regularized_information.shape[0]) * 1e-8
        try:
            edf = float(np.trace(np.linalg.solve(regularized_information, information)))
        except np.linalg.LinAlgError:
            edf = float(np.trace(np.linalg.pinv(regularized_information) @ information))
        aic = -2.0 * fitted["log_likelihood"] + 2.0 * edf
        candidates.append({
            "lambda": float(lam),
            "edf": edf,
            "aic": float(aic),
            **fitted,
        })
    selected = min(candidates, key=lambda item: item["aic"])
    selected["transform"] = transform
    selected["penalty"] = penalty
    selected["lambda_grid"] = [float(value) for value in lambda_grid]
    return selected


# Function guide: _predict_cox_gam is responsible for evaluate a fitted Cox-GAM smooth.
# Inputs: a fitted smooth object and a prediction grid. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _predict_cox_gam(fitted, values):
    """Return the centered Cox-GAM linear predictor on a new grid."""
    return fitted["transform"](values) @ fitted["coefficients"]


# Function guide: _assess_age_gam_linearity is responsible for assess a GAM smooth against a linear Cox term.
# Inputs: the training partition, multiple imputer, and an age transformation. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _assess_age_gam_linearity(df_train, imputer, variable, transformation, transform_function):
    raw_names = list(imputer.feature_names_in_)
    completed_frames = [
        frame.reset_index(drop=True)
        for frame in imputer._completed_frames(df_train[raw_names])
    ]
    outcome = df_train[[TIME_COL, EVENT_COL]].reset_index(drop=True)
    durations = outcome[TIME_COL].to_numpy(dtype=float)
    events = outcome[EVENT_COL].to_numpy(dtype=int)
    checks = []
    for index, completed in enumerate(completed_frames):
        age = completed["age"].to_numpy(dtype=float)
        predictor = np.asarray(transform_function(age), dtype=float)
        item = {
            "variable": variable,
            "transformation": transformation,
            "imputation": index + 1,
            "seed": int(getattr(imputer, "imputation_seeds_", [SEED + index])[index]),
            "method": "Penalized Cox-GAM smooth versus linear Cox",
            "smooth_basis": "Centered P-spline basis with second-difference penalty",
            "n_splines": 10,
            "selection": "Minimum partial-likelihood AIC over lambda grid",
        }
        try:
            linear_fit = _fit_cox_penalized(
                predictor.reshape(-1, 1), durations, events,
            )
            gam_fit = _fit_cox_gam_1d(predictor, durations, events)
            linear_aic = -2.0 * linear_fit["log_likelihood"] + 2.0
            lr = max(
                0.0,
                2.0 * (gam_fit["log_likelihood"] - linear_fit["log_likelihood"]),
            )
            degrees = max(float(gam_fit["edf"] - 1.0), 1e-6)
            p_value = float(stats.chi2.sf(lr, degrees))
            delta_aic = float(linear_aic - gam_fit["aic"])
            nonlinear = p_value < 0.10 and delta_aic > 0.0
            decision = (
                "nonlinear_age_squared" if variable == "age_squared" else "nonlinear_age"
            ) if nonlinear else (
                "linear_age_squared" if variable == "age_squared" else "linear_age"
            )
            item.update({
                "lr": float(lr),
                "df": float(degrees),
                "p": p_value,
                "edf": float(gam_fit["edf"]),
                "lambda": float(gam_fit["lambda"]),
                "aic_linear": float(linear_aic),
                "aic_gam": float(gam_fit["aic"]),
                "delta_aic_linear_minus_gam": delta_aic,
                "decision": decision,
                "p_interpretation": "Approximate likelihood-ratio screen using GAM effective degrees of freedom.",
            })
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            fallback = "linear_age_squared_fallback" if variable == "age_squared" else "linear_age_fallback"
            item.update({"error": str(exc), "decision": fallback})
        checks.append(item)
    valid_p = [
        float(item["p"])
        for item in checks
        if isinstance(item.get("p"), (int, float)) and np.isfinite(item["p"])
    ]
    nonlinear_votes = sum(p < 0.10 for p in valid_p)
    vote_threshold = max(1, int(np.ceil(len(valid_p) / 2.0)))
    nonlinear_label = "nonlinear_age_squared" if variable == "age_squared" else "nonlinear_age"
    linear_label = "linear_age_squared" if variable == "age_squared" else "linear_age"
    summary = {
        "variable": variable,
        "transformation": transformation,
        "method": "Multiple-imputation penalized Cox-GAM smooth versus linear Cox",
        "alpha": 0.10,
        "n_imputations": len(valid_p),
        "p_values": valid_p,
        "n_nonlinear_preferred": int(nonlinear_votes),
        "vote_threshold": int(vote_threshold),
        "decision_rule": "Report a nonlinear GAM shape when at least half of valid imputations have approximate p < 0.10 and GAM AIC is lower.",
        "decision": nonlinear_label if nonlinear_votes >= vote_threshold else linear_label,
        "use_in_final_model": False,
        "note": "Exploratory Cox-GAM diagnostic fitted on completed training datasets; it does not silently replace the deployed age representation.",
    }
    return {"summary": summary, "checks": checks}


# Function guide: assess_age_gam_linearity is responsible for assess raw-age shape with a Cox-GAM smooth.
# Inputs: the training partition and multiple imputer. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def assess_age_gam_linearity(df_train: pd.DataFrame, imputer: MultipleClinicalPreprocessor) -> dict:
    return _assess_age_gam_linearity(
        df_train,
        imputer,
        variable="age",
        transformation="raw age",
        transform_function=lambda age: age,
    )


# Function guide: assess_age_squared_linearity is responsible for assess age-squared shape with a Cox-GAM smooth.
# Inputs: the training partition and multiple imputer. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def assess_age_squared_linearity(
    df_train: pd.DataFrame,
    imputer: MultipleClinicalPreprocessor,
) -> dict:
    """Check the uncentered age-squared transformation with a Cox-GAM smooth."""
    return _assess_age_gam_linearity(
        df_train,
        imputer,
        variable="age_squared",
        transformation="age squared (age ** 2; no centering)",
        transform_function=lambda age: age ** 2,
    )


# Function guide: save_feature_form_diagnostics is responsible for export feature-form diagnostic figures.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def save_feature_form_diagnostics(df_train: pd.DataFrame,
                                  imputer: MultipleClinicalPreprocessor,
                                  output_dir: str | os.PathLike,
                                  age_squared_result: dict | None = None,
                                  age_gam_result: dict | None = None,
                                  age_only: bool = True) -> dict[str, str]:
    """Save an article-style adjusted HR curve for each imputed dataset.

    The article fits ``cph(Surv(...) ~ rcs(age, 3) + covariates)`` and uses
    ``anova`` to report the overall and nonlinear P values.  This figure
    follows its ``Predict(..., fun=exp, ref.zero=TRUE)`` display: the RCS
    curve is plotted as an adjusted hazard ratio with a 95% confidence band,
    and the training-median age is the HR=1 reference.  Only
    ``feature_form_age.png`` is generated by the deployment workflow.
    """
    output_dir = os.fspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    raw_names = list(imputer.feature_names_in_)
    completed_frames = [
        frame.reset_index(drop=True)
        for frame in imputer._completed_frames(df_train[raw_names])
    ]
    # Reset all diagnostic-display indices before concatenating.  The outer
    # split preserves source row labels, whereas the outcome frame is rebuilt
    # with a fresh RangeIndex; positional alignment is required here.
    outcome = df_train[[TIME_COL, EVENT_COL]].reset_index(drop=True)
    raw_age = df_train["age"].to_numpy(dtype=float)
    event_values = outcome[EVENT_COL].to_numpy(dtype=int)
    durations = outcome[TIME_COL].to_numpy(dtype=float)
    events = outcome[EVENT_COL].to_numpy(dtype=int)
    observed_mask = np.isfinite(raw_age)
    paths = {}
    n_imputations = len(completed_frames)
    n_cols = 2
    n_rows = int(np.ceil(n_imputations / n_cols))
    age_checks = {
        int(item["imputation"]): item
        for item in imputer.form_checks_
        if item.get("method") == "Article-style Cox RCS overall and nonlinear Wald tests"
        and item.get("imputation") is not None
    }
    seeds = list(getattr(imputer, "imputation_seeds_", range(SEED, SEED + n_imputations)))

    def _p_value_label(check: dict, key: str) -> str:
        p_value = check.get(key)
        return f"{p_value:.3g}" if isinstance(p_value, (int, float)) else "NA"

    # The primary figure is one row by five columns, with one completed
    # training dataset in each column.  It displays the article-style RCS
    # curve regardless of whether the final common representation is linear;
    # the title records the form selected by the P-value rule.
    comparison_cols = n_imputations
    fig, axes = plt.subplots(
        1, comparison_cols,
        figsize=(3.5 * comparison_cols, 3.5),
        squeeze=False,
        sharey=True,
    )

    def _article_curve(completed: pd.DataFrame, preprocessor: ClinicalPreprocessor):
        """Fit the adjusted RCS Cox model and return HR values on a grid."""
        age = completed["age"].to_numpy(dtype=float)
        rcs = getattr(preprocessor, "age_rcs_diagnostic_", None)
        if rcs is None:
            rcs = RestrictedCubicSpline(n_knots=3).fit(completed[["age"]])
        basis = pd.DataFrame(
            rcs.transform(completed[["age"]]),
            index=completed.index,
            columns=rcs.get_feature_names_out().tolist(),
        )
        covariates = _article_style_covariates(completed, preprocessor.categories_)
        fit_frame = pd.concat([basis, covariates, outcome], axis=1)
        model = CoxPHFitter().fit(fit_frame, TIME_COL, EVENT_COL)

        reference_age = float(np.median(age))
        reference_covariates = {}
        for name in completed.columns:
            if name == "age":
                continue
            if name in _DEPLOYED_CATEGORICAL_FEATURES:
                mode = int(completed[name].mode().iloc[0])
                for level in preprocessor.categories_.get(name, [] )[1:]:
                    reference_covariates[f"{name}_{int(level)}"] = float(mode == level)
            else:
                reference_covariates[name] = float(np.median(completed[name]))

        low, high = np.quantile(age, [0.10, 0.90])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low, high = float(np.min(age)), float(np.max(age))
        grid = np.linspace(float(low), float(high), 200)

        def design(age_values):
            age_values = np.asarray(age_values, dtype=float)
            age_basis = pd.DataFrame(
                rcs.transform(age_values.reshape(-1, 1)),
                columns=rcs.get_feature_names_out().tolist(),
            )
            covariate_frame = pd.DataFrame(
                np.tile(
                    [reference_covariates.get(column, 0.0) for column in covariates.columns],
                    (len(age_values), 1),
                ),
                columns=covariates.columns,
            )
            return pd.concat([age_basis, covariate_frame], axis=1)[model.params_.index]

        grid_design = design(grid).to_numpy(dtype=float)
        reference_design = design([reference_age]).to_numpy(dtype=float)[0]
        differences = grid_design - reference_design
        coefficient = model.params_.loc[model.params_.index].to_numpy(dtype=float)
        covariance = model.variance_matrix_.loc[model.params_.index, model.params_.index].to_numpy(dtype=float)
        log_hr = differences @ coefficient
        variance = np.einsum("ij,jk,ik->i", differences, covariance, differences)
        standard_error = np.sqrt(np.maximum(variance, 0.0))
        return (
            grid,
            np.exp(log_hr),
            np.exp(log_hr - 1.96 * standard_error),
            np.exp(log_hr + 1.96 * standard_error),
            reference_age,
        )

    colors = CELL_COLORS[:n_imputations]
    for index, completed in enumerate(completed_frames):
        color = colors[index]
        ax_raw = axes[0, index]
        try:
            grid, hazard_ratio, lower, upper, reference_age = _article_curve(
                completed,
                imputer.preprocessors_[index],
            )
            ax_raw.fill_between(
                grid,
                lower,
                upper,
                color=color,
                alpha=0.22,
                edgecolor="none",
                label="95% CI" if index == 0 else None,
            )
            ax_raw.plot(
                grid,
                hazard_ratio,
                color=color,
                linewidth=1.0,
                label="RCS Cox" if index == 0 else None,
            )
            ax_raw.axvline(reference_age, color=CELL_COLORS[5], linewidth=0.8, linestyle=":")
        except (ValueError, lifelines.exceptions.ConvergenceError, np.linalg.LinAlgError):
            ax_raw.text(
                0.5, 0.5, "RCS prediction unavailable",
                transform=ax_raw.transAxes, ha="center", va="center",
            )

    for index in range(comparison_cols):
        ax_raw = axes[0, index]
        check = age_checks.get(index + 1, {})
        decision = check.get("decision", "unavailable")
        raw_title = (
            f"Imputation {index + 1}\n"
            f"P={_p_value_label(check, 'p_total')}; "
            f"P nonlinear={_p_value_label(check, 'p_nonlinear')}\n"
            f"selected form: {decision}"
        )
        ax_raw.axhline(1.0, color=CELL_COLORS[5], linewidth=0.8, linestyle="--")
        ax_raw.set_xlabel("Age (years)")
        ax_raw.set_title(raw_title, fontsize=10, fontweight="bold")
        if index == 0:
            ax_raw.set_ylabel("Hazard ratio (95% CI)")
        else:
            ax_raw.set_ylabel("")
        ax_raw.tick_params(direction="out")
        ax_raw.spines[["top", "right", "bottom", "left"]].set_visible(True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(handles, labels, frameon=False, fontsize=7, loc="best")
    fig.tight_layout()
    age_path = os.path.join(output_dir, "feature_form_age.png")
    fig.savefig(age_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths["age"] = os.path.basename(age_path)

    if age_only:
        return paths

    # Figure 2: all imputation-specific Martingale-residual checks.
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(7.0, 3.6 * n_rows), squeeze=False,
    )
    for index, completed in enumerate(completed_frames):
        ax = axes.ravel()[index]
        age_values = completed[["age"]].astype(float)
        completed_age = age_values["age"].to_numpy(dtype=float)
        age_fit = pd.concat([age_values, outcome], axis=1)
        n_bins = min(8, max(3, int(np.unique(completed_age).size)))
        bin_codes = pd.qcut(pd.Series(completed_age), q=n_bins, labels=False, duplicates="drop").to_numpy()
        try:
            residual_model = CoxPHFitter().fit(age_fit, TIME_COL, EVENT_COL)
            residual_frame = residual_model.compute_residuals(age_fit, kind="martingale")
            martingale = residual_frame.loc[age_fit.index, "martingale"].to_numpy(dtype=float)
            finite = np.isfinite(martingale) & np.isfinite(completed_age)
            ax.scatter(
                completed_age[finite], martingale[finite], color=CELL_COLORS[3],
                marker="o", s=12, alpha=0.42, label="Individual residuals",
            )
            residual_x, residual_mean, residual_err = [], [], []
            for code in sorted(pd.Series(bin_codes).dropna().unique().tolist()):
                mask = (bin_codes == code) & finite
                values = martingale[mask]
                if values.size == 0:
                    continue
                residual_x.append(float(np.median(completed_age[mask])))
                residual_mean.append(float(np.mean(values)))
                residual_err.append(
                    1.96 * float(np.std(values, ddof=1) / np.sqrt(values.size))
                    if values.size > 1 else 0.0
                )
            if residual_x:
                ax.errorbar(
                    residual_x, residual_mean, yerr=residual_err,
                    fmt="o", color=CELL_COLORS[2], markersize=4,
                    capsize=3, linewidth=0.8, label="Age-bin residual mean",
                )
            ax.axhline(0.0, color=CELL_COLORS[5], linestyle=":", linewidth=0.8)
        except (ValueError, lifelines.exceptions.ConvergenceError, np.linalg.LinAlgError, KeyError):
            ax.text(0.5, 0.5, "Martingale residuals unavailable",
                    transform=ax.transAxes, ha="center", va="center")
        check = age_checks.get(index + 1, {})
        ax.set_xlabel("Age (years)")
        ax.set_ylabel("Martingale residual")
        ax.set_title(
            f"Imputation {index + 1} (seed={seeds[index]}; {_p_label(index)})",
            fontsize=10, fontweight="bold",
        )
        if index == 0:
            ax.legend(frameon=False, fontsize=7, loc="best")
        ax.tick_params(direction="out")
        ax.spines[["top", "right", "bottom", "left"]].set_visible(True)
    for ax in axes.ravel()[n_imputations:]:
        ax.axis("off")
    fig.tight_layout()
    residual_path = os.path.join(output_dir, "feature_form_age_residuals.png")
    fig.savefig(residual_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths["age_residuals"] = os.path.basename(residual_path)

    # Figure 3: all imputation-specific observed-status displays.  These are
    # descriptive only; censoring-aware evidence is in the Cox panels above.
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(7.0, 3.0 * n_rows), squeeze=False,
    )
    for index, completed in enumerate(completed_frames):
        ax = axes.ravel()[index]
        completed_age = completed["age"].to_numpy(dtype=float)
        rng = np.random.default_rng(SEED + index)
        for event_code, color, label in (
            (0, CELL_COLORS[1], "Censored"),
            (1, CELL_COLORS[0], "Death"),
        ):
            for is_observed, marker, suffix in (
                (True, "o", "observed"),
                (False, "x", "imputed age"),
            ):
                mask = (event_values == event_code) & (observed_mask == is_observed)
                if not mask.any():
                    continue
                y_jitter = event_values[mask] + rng.normal(0.0, 0.035, int(mask.sum()))
                ax.scatter(
                    completed_age[mask], y_jitter, color=color, marker=marker,
                    s=14 if marker == "o" else 20, alpha=0.62,
                    label=f"{label} ({suffix})",
                )
        ax.set_ylim(-0.25, 1.25)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Censored", "Death"])
        ax.set_xlabel("Age (years)")
        ax.set_ylabel("Observed status")
        ax.set_title(
            f"Imputation {index + 1} (seed={seeds[index]}; completed age)",
            fontsize=10, fontweight="bold",
        )
        if index == 0:
            ax.legend(frameon=False, fontsize=7, loc="best")
        ax.tick_params(direction="out")
        ax.spines[["top", "right", "bottom", "left"]].set_visible(True)
    for ax in axes.ravel()[n_imputations:]:
        ax.axis("off")
    fig.tight_layout()
    status_path = os.path.join(output_dir, "feature_form_age_observations.png")
    fig.savefig(status_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths["age_observations"] = os.path.basename(status_path)

    # Figure 4: Martingale residuals for the age-squared linearity check.
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(7.0, 3.6 * n_rows), squeeze=False,
    )
    for index, completed in enumerate(completed_frames):
        ax = axes.ravel()[index]
        age = completed["age"].to_numpy(dtype=float)
        age_squared = age ** 2
        transformed_fit = pd.concat(
            [pd.DataFrame({"age_squared": age_squared}), outcome], axis=1,
        )
        try:
            linear = CoxPHFitter().fit(transformed_fit, TIME_COL, EVENT_COL)
            residual_frame = linear.compute_residuals(transformed_fit, kind="martingale")
            martingale = residual_frame.loc[transformed_fit.index, "martingale"].to_numpy(dtype=float)
            finite = np.isfinite(martingale) & np.isfinite(age_squared)
            ax.scatter(
                age_squared[finite], martingale[finite], color=CELL_COLORS[3],
                marker="o", s=12, alpha=0.42, label="Individual residuals",
            )
            n_bins = min(8, max(3, int(np.unique(age_squared).size)))
            bin_codes = pd.qcut(pd.Series(age_squared), q=n_bins, labels=False, duplicates="drop").to_numpy()
            residual_x, residual_mean, residual_err = [], [], []
            for code in sorted(pd.Series(bin_codes).dropna().unique().tolist()):
                mask = (bin_codes == code) & finite
                values = martingale[mask]
                if values.size == 0:
                    continue
                residual_x.append(float(np.median(age_squared[mask])))
                residual_mean.append(float(np.mean(values)))
                residual_err.append(
                    1.96 * float(np.std(values, ddof=1) / np.sqrt(values.size))
                    if values.size > 1 else 0.0
                )
            if residual_x:
                ax.errorbar(
                    residual_x, residual_mean, yerr=residual_err,
                    fmt="o", color=CELL_COLORS[2], markersize=4,
                    capsize=3, linewidth=0.8, label="Age-squared-bin residual mean",
                )
            ax.axhline(0.0, color=CELL_COLORS[5], linestyle=":", linewidth=0.8)
        except (ValueError, lifelines.exceptions.ConvergenceError, np.linalg.LinAlgError, KeyError):
            ax.text(0.5, 0.5, "Martingale residuals unavailable",
                    transform=ax.transAxes, ha="center", va="center")
        check = squared_checks.get(index + 1, {})
        p_value = check.get("p")
        p_label = f"LRT p={p_value:.3f}" if isinstance(p_value, (int, float)) else "LRT unavailable"
        ax.set_xlabel(r"Age squared (years$^2$)")
        ax.set_ylabel("Martingale residual")
        ax.set_title(
            f"Imputation {index + 1} ({p_label})", fontsize=10, fontweight="bold",
        )
        if index == 0:
            ax.legend(frameon=False, fontsize=7, loc="best")
        ax.tick_params(direction="out")
        ax.spines[["top", "right", "bottom", "left"]].set_visible(True)
    for ax in axes.ravel()[n_imputations:]:
        ax.axis("off")
    fig.tight_layout()
    squared_residual_path = os.path.join(output_dir, "feature_form_age_squared_residuals.png")
    fig.savefig(squared_residual_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths["age_squared_residuals"] = os.path.basename(squared_residual_path)

    # Use the first completed dataset for the categorical coefficient audit;
    # the categorical representation is identical across all imputations.
    completed = completed_frames[0]

    # Categorical dummy-coded coefficient audit.  This is not a linearity
    # diagnostic: dummy-coded categories do not require a linearity check.
    categorical = [name for name in raw_names if name in _DEPLOYED_CATEGORICAL_FEATURES]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), squeeze=False)
    for ax, name in zip(axes.ravel(), categorical):
        values = completed[name].astype(int)
        levels = sorted(values.unique().tolist())
        counts = values.value_counts().reindex(levels, fill_value=0)
        dummy = pd.get_dummies(values, prefix=name, drop_first=True, dtype=float)
        fit_frame = pd.concat([dummy.reset_index(drop=True), outcome.reset_index(drop=True)], axis=1)
        estimates = {levels[0]: (0.0, 0.0, 0.0)}
        if dummy.shape[1] > 0:
            try:
                model = CoxPHFitter().fit(fit_frame, TIME_COL, EVENT_COL)
                ci = model.confidence_intervals_
                for level in levels[1:]:
                    column = f"{name}_{int(level)}"
                    if column not in model.params_.index:
                        estimates[level] = (0.0, 0.0, 0.0)
                        continue
                    estimate = float(model.params_.loc[column])
                    lower = float(ci.loc[column].iloc[0])
                    upper = float(ci.loc[column].iloc[1])
                    estimates[level] = (estimate, lower, upper)
            except (ValueError, lifelines.exceptions.ConvergenceError, np.linalg.LinAlgError):
                pass
        y_pos = np.arange(len(levels))
        for position, level in enumerate(levels):
            estimate, lower, upper = estimates.get(level, (0.0, 0.0, 0.0))
            if level == levels[0]:
                ax.plot(estimate, position, marker="s", color=CELL_COLORS[3], markersize=4)
            else:
                ax.errorbar(
                    estimate, position,
                    xerr=[[estimate - lower], [upper - estimate]],
                    fmt="o", color=CELL_COLORS[0], markersize=4,
                    capsize=3, linewidth=0.8,
                )
        ax.axvline(0.0, color=CELL_COLORS[5], linestyle=":", linewidth=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{int(level)} (n={int(counts.loc[level])})" for level in levels])
        ax.set_xlabel("Log hazard ratio")
        ax.set_title(f"{name}: treatment-coded dummies")
        ax.tick_params(direction="out")
        ax.spines[["top", "right", "bottom", "left"]].set_visible(True)
    fig.tight_layout()
    categorical_path = os.path.join(output_dir, "categorical_dummy_encoding.png")
    fig.savefig(categorical_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths["categorical_encoding"] = os.path.basename(categorical_path)
    return paths


# Class guide: ExponentialAFTFitter is responsible for perform the requested operation.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
class ExponentialAFTFitter(ParametricRegressionFitter):
    """Exponential AFT regression with a fixed Weibull shape of one.

    Lifelines provides Weibull, log-normal, and log-logistic AFT fitters, but
    its exponential regression form is not available in every supported
    version.  This small adapter supplies the cumulative-hazard equation and
    the prediction methods needed by the distribution-selection workflow.
    """

    _fitted_parameter_names = ["lambda_"]

    # Function guide: _cumulative_hazard is responsible for perform hazard.
    # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def _cumulative_hazard(self, params, T, Xs):
        # In an exponential AFT model, the covariates scale time through the
        # exponentiated linear predictor.  Lifelines differentiates this
        # expression when estimating the regression parameters.
        return T / anp.exp(anp.dot(Xs["lambda_"], params["lambda_"]))

    # Function guide: fit is responsible for fit the requested operation.
    # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def fit(self, df, duration_col, event_col=None, **kwargs):
        # Regress the log time scale on an intercept plus every supplied feature;
        # duration and event columns are never treated as predictors.
        features = [c for c in df.columns if c not in (duration_col, event_col)]
        return super().fit(
            df, duration_col, event_col,
            regressors={"lambda_": "1 + " + " + ".join(features)}, **kwargs,
        )

    # Function guide: predict_percentile is responsible for perform percentile.
    # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def predict_percentile(self, df, *, p=0.5, conditional_after=None):
        # At one unit of time, the fitted cumulative hazard is the individual
        # hazard rate.  Solving H(t) = -log(p) yields the requested percentile.
        if not 0 < p < 1:
            raise ValueError("Survival percentile must be strictly between zero and one.")
        unit_hazard = self.predict_cumulative_hazard(df, times=[1.0]).iloc[0]
        return -np.log(p) / unit_hazard

    # Function guide: predict_expectation is responsible for perform expectation.
    # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def predict_expectation(self, df):
        # For an exponential distribution the expected time is the reciprocal
        # of the individual unit-time hazard.
        return 1.0 / self.predict_cumulative_hazard(df, times=[1.0]).iloc[0]

_UNIVAR_FITTERS = {
    "Weibull":     WeibullFitter,
    "Log-Normal":  LogNormalFitter,
    "Log-Logistic":LogLogisticFitter,
    "Exponential": ExponentialFitter,
}

_AFT_CLASSES = {
    "Weibull":     WeibullAFTFitter,
    "Log-Normal":  LogNormalAFTFitter,
    "Log-Logistic":LogLogisticAFTFitter,
    "Exponential": ExponentialAFTFitter,
}

# Function guide: test_distributions is responsible for test distributions.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def test_distributions(df: pd.DataFrame) -> dict:
    """
    Fit 4 parametric distributions to observed survival times (with censoring).
    Compare AIC / BIC to select the AFT family for the parametric model.

    Returns
    -------
    dict with keys:
        "table"        : pd.DataFrame with AIC/BIC/median per distribution
        "best"         : name of best distribution (lowest AIC)
        "aft_class"    : the lifelines AFT fitter class to use
        "fitters"      : fitted univariate fitter objects (for plotting)
    """
    # Fit each candidate to the marginal observed-time distribution while
    # passing the event indicator so censored observations contribute correctly
    # to the likelihood without being treated as deaths.
    rows, fitters = [], {}
    for name, cls in _UNIVAR_FITTERS.items():
        f = cls()
        f.fit(df[TIME_COL], event_observed=df[EVENT_COL], label=name)
        fitters[name] = f
        rows.append({"Distribution": name,
                      "AIC": float(f.AIC_),
                      "BIC": float(f.BIC_),
                      "Median (months)": round(float(f.median_survival_time_), 1)})

    # AIC is the primary selection criterion; BIC and the fitted median are
    # retained for inspection in the model report.
    table = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
    best  = table.iloc[0]["Distribution"]
    return {"table": table, "best": best,
            "aft_class": _AFT_CLASSES[best], "fitters": fitters}


# ── 5a. Cox Proportional Hazards (lifelines, L2-regularised) ─────────────────

# Function guide: tune_cox_penalizer is responsible for select the Cox penalizer by cross-validation.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def tune_cox_penalizer(df_train: pd.DataFrame, n_folds: int = 5,
                        feat_df: pd.DataFrame = None,
                        raw_feat_df: pd.DataFrame = None,
                        diagnostics: list | None = None,
                        fold_imputation: str = "median_mode") -> float:
    """Select the Cox L2 penalizer with stratified training-only CV.

    The candidate grid follows the project specification.  When raw features
    are supplied, ``ClinicalPreprocessor`` is refit inside each fold so
    imputation, spline knots, and dummy-level decisions cannot use validation
    rows.  The fold score is the lifelines concordance index, and a candidate
    is accepted only when every fold completes successfully.
    """
    if feat_df is None:
        feat_df = preprocess(df_train)
    df_tr = feat_df.copy()
    df_tr[TIME_COL]  = df_train[TIME_COL].values
    df_tr[EVENT_COL] = df_train[EVENT_COL].values

    # Build the fold-specific datasets before comparing penalties.  This keeps
    # every penalty on the same preprocessing realization within a fold.
    folds = []
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    for fold_number, (tr_idx, va_idx) in enumerate(
        kf.split(df_tr, df_train[EVENT_COL].values)
    ):
        fold_tr, fold_va = df_tr.iloc[tr_idx].copy(), df_tr.iloc[va_idx].copy()
        if raw_feat_df is not None:
            if fold_imputation == "mice":
                imputer = MultipleClinicalPreprocessor(
                    n_imputations=2, random_state=SEED + fold_number,
                ).fit(
                    raw_feat_df.iloc[tr_idx],
                    df_train.iloc[tr_idx][[TIME_COL, EVENT_COL]],
                )
                fold_tr = imputer.transform_multiple(raw_feat_df.iloc[tr_idx])[0]
                fold_va = imputer.transform_multiple(raw_feat_df.iloc[va_idx])[0]
            else:
                imputer = ClinicalPreprocessor().fit(raw_feat_df.iloc[tr_idx], df_train.iloc[tr_idx])
                fold_tr = imputer.transform(raw_feat_df.iloc[tr_idx])
                fold_va = imputer.transform(raw_feat_df.iloc[va_idx])
            for frame, indices in ((fold_tr, tr_idx), (fold_va, va_idx)):
                frame[TIME_COL] = df_train.iloc[indices][TIME_COL].to_numpy()
                frame[EVENT_COL] = df_train.iloc[indices][EVENT_COL].to_numpy()
        folds.append((fold_tr, fold_va))

    best_c, best_pen = -1.0, None
    for pen in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
        scores = []
        failures = []
        # Evaluate one penalty on all folds before comparing it with the best
        # candidate.  No test rows participate in this search.
        for fold_tr, fold_va in folds:
            try:
                m = CoxPHFitter(penalizer=pen, l1_ratio=0.0)
                m.fit(fold_tr, duration_col=TIME_COL, event_col=EVENT_COL)
                scores.append(
                    m.score(
                        fold_va,
                        scoring_method="concordance_index",
                    )
                )
            except (ValueError, lifelines.exceptions.ConvergenceError) as exc:
                failures.append(str(exc))
        if diagnostics is not None:
            diagnostics.append({"penalizer": pen, "fold_scores": scores, "failures": failures})
        if len(scores) == n_folds and np.mean(scores) > best_c:
            best_c, best_pen = np.mean(scores), pen
    if best_pen is None:
        raise RuntimeError("No Cox penalizer completed every CV fold successfully.")
    return best_pen


# Function guide: build_cox is responsible for fit the Cox proportional hazards model.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def build_cox(df_train: pd.DataFrame, penalizer: Optional[float] = None,
              feat_df: pd.DataFrame = None,
              raw_feat_df: pd.DataFrame = None,
              fold_imputation: str = "median_mode") -> CoxPHFitter:
    """Fit the deployable L2-regularized Cox proportional-hazards model.

    If no penalizer is supplied, it is selected by the training-only CV helper.
    The returned lifelines object contains the fitted coefficients and Breslow
    baseline hazard needed to obtain individual risk scores and survival
    curves.  The two private attributes preserve the selected penalty and its
    audit trail for the application bundle.
    """
    if feat_df is None:
        feat_df = preprocess(df_train)
    diagnostics = []
    if penalizer is None:
        penalizer = tune_cox_penalizer(
            df_train, feat_df=feat_df, raw_feat_df=raw_feat_df,
            diagnostics=diagnostics, fold_imputation=fold_imputation,
        )
    # Append survival outcomes only for fitting.  The feature frame itself
    # remains free of duration/event columns for prediction calls.
    df_fit = feat_df.copy()
    df_fit[TIME_COL]  = df_train[TIME_COL].values
    df_fit[EVENT_COL] = df_train[EVENT_COL].values
    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=0.0)
    cph.fit(df_fit, duration_col=TIME_COL, event_col=EVENT_COL)
    cph._penalizer_used = penalizer
    cph._cv_diagnostics = diagnostics
    return cph


# ── 5b. Parametric AFT (best distribution by AIC) ────────────────────────────

# Function guide: build_aft is responsible for fit the parametric accelerated failure-time model.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def build_aft(df_train: pd.DataFrame, aft_class=None,
              penalizer: float = 0.01,
              feat_df: pd.DataFrame = None) -> object:
    """Fit the selected parametric accelerated-failure-time model.

    ``aft_class`` is normally chosen by ``test_distributions`` using the
    lowest marginal AIC.  The default Weibull class keeps this factory useful
    for tests and explicit model comparisons.  The penalizer is passed to the
    fitter and the event indicator is supplied so right-censored observations
    contribute to the likelihood correctly.
    """
    if aft_class is None:
        aft_class = WeibullAFTFitter
    if feat_df is None:
        feat_df = preprocess(df_train)
    # Keep the input feature order and append outcomes only at fit time.
    df_fit = feat_df.copy()
    df_fit[TIME_COL]  = df_train[TIME_COL].values
    df_fit[EVENT_COL] = df_train[EVENT_COL].values
    aft = aft_class(penalizer=penalizer)
    aft.fit(df_fit, duration_col=TIME_COL, event_col=EVENT_COL)
    return aft


# ── 5c. Random Survival Forest (sksurv + Optuna) ─────────────────────────────

# Function guide: build_rsf is responsible for fit the random survival forest model.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def build_rsf(df_train: pd.DataFrame, n_trials: int = 40,
              feat_df: pd.DataFrame = None) -> tuple:
    """
    Tune a research-only random survival forest with Optuna.

    The TPE sampler explores tree depth, node-size, feature-subsampling, and
    ensemble-size settings.  Each trial is scored with five-fold concordance
    index on the supplied training partition.  This utility is retained for
    experimentation and is not included in the current deployed bundle.

    Returns (fitted_rsf, best_params, best_cv_cindex)
    """
    import optuna
    from sksurv.ensemble import RandomSurvivalForest

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if feat_df is None:
        feat_df = preprocess(df_train)
    # Convert once to a numeric matrix and build a structured survival target
    # for scikit-survival's estimator interface.
    X = feat_df.values.astype(float)
    y = Surv.from_arrays(event=df_train[EVENT_COL].astype(bool).values,
                         time=df_train[TIME_COL].values,
                         name_event=EVENT_COL, name_time=TIME_COL)
    feat_cols = feat_df.columns.tolist()

    # Function guide: objective is responsible for perform the requested operation.
    # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def objective(trial: optuna.Trial) -> float:
        # Every Optuna trial receives a fresh five-fold CV evaluation so the
        # hyperparameter score is not based on the final refit predictions.
        params = dict(
            n_estimators  = trial.suggest_int("n_estimators", 100, 500, step=50),
            max_depth     = trial.suggest_int("max_depth", 3, 12),
            min_samples_split = trial.suggest_int("min_samples_split", 5, 40),
            min_samples_leaf  = trial.suggest_int("min_samples_leaf", 3, 20),
            max_features  = trial.suggest_float("max_features", 0.3, 1.0),
            n_jobs=-1, random_state=SEED,
        )
        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
        c_scores = []
        for tr_idx, va_idx in kf.split(X):
            rsf = RandomSurvivalForest(**params)
            rsf.fit(X[tr_idx], y[tr_idx])
            c_scores.append(rsf.score(X[va_idx], y[va_idx]))
        return float(np.mean(c_scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params | dict(n_jobs=-1, random_state=SEED)
    rsf = RandomSurvivalForest(**best)
    rsf.fit(X, y)
    rsf._feat_cols = feat_cols    # store for prediction helper
    return rsf, study.best_params, study.best_value


# ── 5d. DeepSurv (pycox CoxPH + PyTorch MLP + Optuna) ────────────────────────

if _DEEPSURV_AVAILABLE:
    # Class guide: _MLP is responsible for perform the requested operation.
    # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    class _MLP(nn.Module):
        """Small feed-forward risk network used by the research DeepSurv path.

        Each hidden block normalizes activations, applies a ReLU nonlinearity,
        and uses dropout for regularization.  The final single output is a
        learned log-risk score rather than a direct survival probability.
        """
        # Function guide: __init__ is responsible for initialize object state.
        # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
        # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
        def __init__(self, in_features: int, num_nodes: list[int], dropout: float):
            super().__init__()
            # Build a configurable stack while keeping the final output scalar
            # so pycox can optimize the Cox partial-likelihood objective.
            layers: list[nn.Module] = []
            prev = in_features
            for h in num_nodes:
                layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                            nn.ReLU(), nn.Dropout(dropout)]
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)

        # Function guide: forward is responsible for perform the requested operation.
        # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
        # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
        def forward(self, x):
            return self.net(x)

    # Function guide: _make_mlp is responsible for create mlp.
    # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def _make_mlp(in_features: int, num_nodes: list[int], dropout: float) -> "_MLP":
        return _MLP(in_features, num_nodes, dropout)


# Class guide: DeepSurvWrapper is responsible for perform the requested operation.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
class DeepSurvWrapper:
    """
    Serialisable wrapper around pycox CoxPH.

    Stores network state dict + baseline hazards as plain Python objects
    so the wrapper can be round-tripped through joblib without needing
    a live torch/pycox session at load time.
    """

    # Function guide: __init__ is responsible for initialize object state.
    # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def __init__(self):
        self.scaler            = None
        self._feat_cols: list  = []
        self._net_params: tuple = ()    # (in_features, num_nodes, dropout)
        self._net_state: dict  = {}
        self._baseline_haz     = None   # pd.Series  (index=time, values=hazard)
        self._baseline_cumhaz  = None   # pd.Series
        self.train_params: dict = {}

    # ── Training ─────────────────────────────────────────────────────────────

    # Function guide: fit is responsible for fit the requested operation.
    # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def fit(self, df_train: pd.DataFrame, params: dict,
            epochs: int = 150, feat_df: pd.DataFrame = None) -> "DeepSurvWrapper":
        """Fit the neural Cox model and retain serializable prediction state.

        The scaler, network architecture, network weights, and baseline
        hazards are copied into plain attributes.  This makes the wrapper
        reloadable from joblib without requiring the original training
        dataloader or optimizer object.  DeepSurv remains an optional research
        comparison and is not part of the current deployment bundle.
        """
        if not _DEEPSURV_AVAILABLE:
            raise ImportError("DeepSurv requires torch + pycox.")
        if feat_df is None:
            feat_df = preprocess(df_train)

        # Store the feature schema before converting to tensors so prediction
        # can reject/reorder data consistently after deserialization.
        self._feat_cols = feat_df.columns.tolist()
        X = feat_df.values.astype("float32")
        T = df_train[TIME_COL].values.astype("float32")
        E = df_train[EVENT_COL].values.astype("float32")

        # Neural optimization is sensitive to feature scale; fit the scaler on
        # training rows and reuse it unchanged for validation and prediction.
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X).astype("float32")

        num_nodes = [params["n_nodes"]] * params["n_layers"]
        dropout   = params["dropout"]
        self._net_params = (Xs.shape[1], num_nodes, dropout)
        self.train_params = params

        torch.manual_seed(SEED)
        net = _make_mlp(*self._net_params)
        # Pass class (not instance) so pycox binds net.parameters() automatically
        model = _PycoxCoxPH(net, tt.optim.Adam)
        model.optimizer.set_lr(params["lr"])

        dl = model.make_dataloader((Xs, (T, E)),
                                    batch_size=params["batch_size"],
                                    shuffle=True)
        model.fit_dataloader(
            dl, epochs=epochs,
            callbacks=[tt.callbacks.EarlyStopping(patience=20)],
            verbose=False,
        )
        model.compute_baseline_hazards(input=Xs, target=(T, E))

        # Verify sign convention: if predictions are inverted (C < 0.48),
        # negate the output layer weights so higher score = more risk.  The
        # survival metrics in this module assume that risk scores increase with
        # event hazard.
        with torch.no_grad():
            log_h_tr = model.predict(Xs).flatten()
        c_tr, *_ = concordance_index_censored(E.astype(bool), T, log_h_tr)
        if c_tr < 0.48:
            out_layer = net.net[-1]   # final nn.Linear(hidden, 1)
            with torch.no_grad():
                out_layer.weight.data.mul_(-1)
                if out_layer.bias is not None:
                    out_layer.bias.data.mul_(-1)
            model.compute_baseline_hazards(input=Xs, target=(T, E))

        self._net_state    = {k: v.cpu().clone() for k, v in net.state_dict().items()}
        self._baseline_haz    = model.baseline_hazards_.copy()
        self._baseline_cumhaz = model.baseline_cumulative_hazards_.copy()
        return self

    # ── Reconstruction (after deserialisation) ───────────────────────────────

    # Function guide: _model is responsible for perform the requested operation.
    # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def _model(self) -> "_PycoxCoxPH":
        """Reconstruct a prediction-only pycox model from saved state."""
        # Recreate the same network shape, restore its weights, and attach the
        # stored baseline hazards required for survival-curve prediction.
        net = _make_mlp(*self._net_params)
        net.load_state_dict(self._net_state)
        net.eval()
        m = _PycoxCoxPH(net, tt.optim.Adam())
        m.baseline_hazards_             = self._baseline_haz
        m.baseline_cumulative_hazards_   = self._baseline_cumhaz
        return m

    # ── Prediction ───────────────────────────────────────────────────────────

    # Function guide: predict_risk is responsible for perform risk.
    # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def predict_risk(self, feat_df: pd.DataFrame) -> np.ndarray:
        """Return log partial hazard, where larger values indicate more risk."""
        # Select the fitted schema before scaling so column order is stable.
        Xs = self.scaler.transform(feat_df[self._feat_cols].values).astype("float32")
        with torch.no_grad():
            return self._model().predict(Xs).flatten()

    # Function guide: predict_surv_df is responsible for perform surv df.
    # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def predict_surv_df(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        """Return survival probabilities indexed by time and patient."""
        Xs = self.scaler.transform(feat_df[self._feat_cols].values).astype("float32")
        return self._model().predict_surv_df(Xs)


# Function guide: tune_deepsurv is responsible for tune deepsurv.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def tune_deepsurv(df_train: pd.DataFrame, n_trials: int = 15,
                  val_frac: float = 0.20,
                  feat_df: pd.DataFrame = None) -> tuple:
    """
    Optuna search for DeepSurv hyperparameters using validation C-index.

    This optional research path uses the final fraction of the supplied
    training partition as a temporary validation set.  It is separate from
    the fixed deployment train/test evaluation and should not be interpreted
    as an external validation result.

    Returns (best_params, best_cindex)
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if not _DEEPSURV_AVAILABLE:
        raise ImportError("DeepSurv requires torch + pycox.")
    if feat_df is None:
        feat_df = preprocess(df_train)

    n_val = max(50, int(len(df_train) * val_frac))
    df_tr  = df_train.iloc[:-n_val].copy()
    df_va  = df_train.iloc[-n_val:].copy()
    ftr    = feat_df.iloc[:-n_val].reset_index(drop=True)
    fva    = feat_df.iloc[-n_val:].reset_index(drop=True)

    e_va = df_va[EVENT_COL].astype(bool).values
    t_va = df_va[TIME_COL].values

    # Function guide: objective is responsible for perform the requested operation.
    # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_layers   = trial.suggest_int("n_layers", 1, 3),
            n_nodes    = trial.suggest_int("n_nodes", 32, 128, step=32),
            dropout    = trial.suggest_float("dropout", 0.0, 0.3),
            lr         = trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            batch_size = trial.suggest_categorical("batch_size", [64, 128, 256]),
        )
        try:
            w = DeepSurvWrapper().fit(df_tr, params, epochs=150, feat_df=ftr)
            risk = w.predict_risk(fva)
            if e_va.sum() < 3:
                return 0.5
            c, *_ = concordance_index_censored(e_va, t_va, risk)
            return float(c)
        except Exception:
            return 0.5

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


# Function guide: build_deepsurv is responsible for fit the DeepSurv model.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def build_deepsurv(df_train: pd.DataFrame, params: dict,
                   epochs: int = 200,
                   feat_df: pd.DataFrame = None) -> DeepSurvWrapper:
    return DeepSurvWrapper().fit(df_train, params, epochs=epochs, feat_df=feat_df)


# ── 6. Unified evaluation ────────────────────────────────────────────────────

_EVAL_TIMES = np.array([12.0, 24.0, 36.0, 48.0, 60.0])


# Function guide: _surv_matrix_lifelines is responsible for perform matrix lifelines.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _surv_matrix_lifelines(model, feat_df: pd.DataFrame,
                            times: np.ndarray) -> np.ndarray:
    """Return a common ``(n_samples, n_times)`` matrix from lifelines.

    Lifelines returns a DataFrame indexed by its internal event-time grid with
    one column per patient.  The evaluation metrics require one row per
    patient and one column per requested horizon, so each patient curve is
    interpolated onto the shared horizon vector and then transposed.
    """
    # shape from lifelines: (n_internal_times, n_samples)
    sf = model.predict_survival_function(feat_df)
    t_axis = sf.index.values
    # Interpolate each patient's survival function at the exact evaluation
    # horizons.  Survival is one before the first internal time and held at
    # the last available value after the final internal time.
    mat = np.column_stack([
        np.interp(times, t_axis, sf.iloc[:, i].values,
                  left=1.0, right=float(sf.iloc[-1, i]))
        for i in range(sf.shape[1])
    ]).T   # (n_samples, n_times)
    return mat.clip(0.0, 1.0)


# Function guide: _surv_matrix_sksurv is responsible for perform matrix sksurv.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _surv_matrix_sksurv(surv_fns, times: np.ndarray) -> np.ndarray:
    """Evaluate scikit-survival step functions on the common time grid."""
    # Each returned step function belongs to one patient; row_stack preserves
    # the same patient-by-horizon orientation used for lifelines models.
    return np.row_stack([
        np.array([fn(t) for t in times]) for fn in surv_fns
    ]).clip(0.0, 1.0)


# Function guide: _surv_matrix_deepsurv is responsible for perform matrix deepsurv.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _surv_matrix_deepsurv(wrapper: DeepSurvWrapper, feat_df: pd.DataFrame,
                           times: np.ndarray) -> np.ndarray:
    """Interpolate DeepSurv survival curves onto the evaluation horizons."""
    # DeepSurv exposes the same time-by-patient layout as lifelines, so reuse
    # the interpolation convention while keeping this helper explicit.
    sf = wrapper.predict_surv_df(feat_df)   # (n_internal_times, n_samples)
    t_axis = sf.index.values
    mat = np.column_stack([
        np.interp(times, t_axis, sf.iloc[:, i].values,
                  left=1.0, right=float(sf.iloc[-1, i]))
        for i in range(sf.shape[1])
    ]).T
    return mat.clip(0.0, 1.0)


# Function guide: _clip_times is responsible for perform times.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _clip_times(times: np.ndarray, y_train, y_test) -> np.ndarray:
    """Keep horizons strictly inside the common observed follow-up range.

    IPCW-based metrics cannot evaluate a horizon beyond the support available
    in the training censoring distribution.  Restricting times to the shared
    interior of train and test avoids optimistic extrapolation and makes the
    same requested horizons valid for both partitions.
    """
    times = np.unique(np.asarray(times, dtype=float))
    t_lo = max(y_train[TIME_COL].min(), y_test[TIME_COL].min())
    t_hi = min(y_train[TIME_COL].max(), y_test[TIME_COL].max())
    return times[(times > t_lo) & (times < t_hi)]


# Function guide: administrative_censor is responsible for perform censor.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def administrative_censor(y, times):
    """Censor beyond the evaluation horizon without changing earlier outcomes.

    sksurv estimates IPCW for observed event times, including events later than
    every requested horizon. Capping follow-up just after the final horizon
    prevents irrelevant late events from exceeding training censoring support.
    """
    # Censor records just after the last requested horizon.  Events before the
    # horizon remain unchanged; later events become administratively censored
    # so they cannot extend the IPCW support beyond the evaluation window.
    if len(times) == 0:
        raise ValueError("No evaluation times lie inside both follow-up ranges.")
    tau = np.nextafter(float(max(times)), np.inf)
    result = y.copy()
    beyond = result[TIME_COL] > tau
    result[EVENT_COL][beyond] = False
    result[TIME_COL][beyond] = tau
    return result


# Function guide: evaluate is responsible for evaluate the requested operation.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def evaluate(model, model_type: str,
             df_test: pd.DataFrame,
             y_train, y_test,
             times: np.ndarray = _EVAL_TIMES,
             feat_df_test: pd.DataFrame = None,
             n_boot: int = 300) -> dict:
    """Evaluate a survival model on a held-out test cohort.

    The routine computes a risk score and survival-probability matrix first,
    then derives three complementary metrics: a bootstrap confidence interval
    for the concordance index, an IPCW-corrected integrated Brier score, and
    cumulative/dynamic time-dependent AUC.  Training outcomes are used only to
    estimate censoring weights; all reported test predictions come from the
    supplied model and test features.

    Parameters
    ----------
    model_type    : "cox" | "aft" | "rsf" | "deepsurv"
    feat_df_test  : pre-processed feature matrix (rows = patients, cols = features).
                    If None, falls back to preprocess(df_test) for simulated data.
    y_train, y_test : sksurv structured arrays (field names = TIME_COL / EVENT_COL)
    """
    # Use only horizons supported by both partitions and cap test follow-up for
    # IPCW calculations at the final horizon.
    times = _clip_times(times, y_train, y_test)
    if len(times) < 2:
        raise ValueError("At least two supported horizons are required for IBS.")
    y_ipcw = administrative_censor(y_test, times)
    e_te  = y_test[EVENT_COL].astype(bool)
    t_te  = y_test[TIME_COL].astype(float)

    # Resolve the feature matrix.  Production callers pass the training-fitted
    # preprocessing output explicitly; the fallback exists for synthetic tests.
    if feat_df_test is None:
        feat_df_test = preprocess(df_test)

    # Build the model-specific risk score and put all survival predictions into
    # the same (patient, horizon) matrix layout.
    if model_type in ("cox", "aft"):
        if model_type == "cox":
            risk = model.predict_partial_hazard(feat_df_test).values.astype(float)
        else:
            risk = (1.0 / model.predict_median(feat_df_test).values).astype(float)
        surv_mat = _surv_matrix_lifelines(model, feat_df_test, times)

    elif model_type == "rsf":
        X = feat_df_test.values.astype(float)
        risk = model.predict(X).astype(float)
        surv_mat = _surv_matrix_sksurv(
            model.predict_survival_function(X), times)

    elif model_type == "deepsurv":
        risk = model.predict_risk(feat_df_test).astype(float)
        surv_mat = _surv_matrix_deepsurv(model, feat_df_test, times)

    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    # The C-index evaluates ranking of risk under censoring.  Bootstrap rows
    # with replacement to quantify test-set sampling uncertainty; resamples
    # with fewer than two events cannot provide a stable concordance estimate.
    c_idx, *_ = concordance_index_censored(e_te, t_te, risk)

    rng = np.random.default_rng(SEED)
    n = len(df_test)
    boot_c = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        eb, tb, rb = e_te[idx], t_te[idx], risk[idx]
        if eb.sum() < 2:
            continue
        try:
            cb, *_ = concordance_index_censored(eb, tb, rb)
            boot_c.append(cb)
        except Exception:
            pass
    ci_lo, ci_hi = np.percentile(boot_c, [2.5, 97.5]) if boot_c else (np.nan, np.nan)

    # IBS evaluates the squared error of the full survival curve over time and
    # uses the training censoring distribution for inverse-probability weights.
    metric_errors = {}
    try:
        ibs = float(integrated_brier_score(y_train, y_ipcw, surv_mat, times))
    except ValueError as exc:
        ibs = np.nan
        metric_errors["ibs"] = str(exc)

    # Dynamic AUC measures discrimination separately at each horizon and then
    # reports the integrated/mean value returned by scikit-survival.
    try:
        auc_vals, mean_auc = cumulative_dynamic_auc(y_train, y_ipcw, risk, times)
        auc_vals  = auc_vals.tolist()
        mean_auc  = float(mean_auc)
    except ValueError as exc:
        auc_vals  = [np.nan] * len(times)
        mean_auc  = np.nan
        metric_errors["auc"] = str(exc)

    return dict(
        c_index    = float(c_idx),
        ci_lo      = float(ci_lo),
        ci_hi      = float(ci_hi),
        ibs        = ibs,
        auc_vals   = auc_vals,
        mean_auc   = mean_auc,
        surv_mat   = surv_mat,
        risk       = risk,
        times      = times.tolist(),
        metric_errors = metric_errors,
        bootstrap_successful = len(boot_c),
    )


# Function guide: evaluate_train is responsible for evaluate train.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def evaluate_train(model, model_type: str,
                   df_train: pd.DataFrame,
                   y_train,
                   times: np.ndarray = _EVAL_TIMES,
                   feat_df_train: pd.DataFrame = None) -> dict:
    """Compute apparent training C-index and IBS for gap assessment.

    Training metrics intentionally do not receive a bootstrap confidence
    interval.  They describe apparent performance on rows used for fitting and
    should be displayed beside the independent test metrics so a train/test
    gap remains visible.
    """
    e_tr = y_train[EVENT_COL].astype(bool)
    t_tr = y_train[TIME_COL].astype(float)

    # Reuse the caller-supplied training feature frame whenever possible so
    # evaluation cannot silently refit preprocessing on a different sample.
    if feat_df_train is None:
        feat_df_train = preprocess(df_train)

    if model_type == "cox":
        risk = model.predict_partial_hazard(feat_df_train).values.astype(float)
    elif model_type == "aft":
        risk = (1.0 / model.predict_median(feat_df_train).values).astype(float)
    elif model_type == "rsf":
        X = feat_df_train.values.astype(float)
        risk = model.predict(X).astype(float)
    elif model_type == "deepsurv":
        risk = model.predict_risk(feat_df_train).astype(float)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    # Higher risk must mean shorter survival.  The model-specific score
    # conversions above align Cox, AFT, RSF, and DeepSurv with that convention.
    c_idx, *_ = concordance_index_censored(e_tr, t_tr, risk)

    times_tr = _clip_times(times, y_train, y_train)

    ibs = np.nan
    if len(times_tr) >= 2:
        if model_type in ("cox", "aft"):
            surv_mat = _surv_matrix_lifelines(model, feat_df_train, times_tr)
        elif model_type == "rsf":
            surv_mat = _surv_matrix_sksurv(
                model.predict_survival_function(feat_df_train.values.astype(float)),
                times_tr)
        elif model_type == "deepsurv":
            surv_mat = _surv_matrix_deepsurv(model, feat_df_train, times_tr)
        ibs = float(integrated_brier_score(y_train, y_train, surv_mat, times_tr))

    return dict(c_index=float(c_idx), ibs=ibs, times=times_tr.tolist())


# Function guide: risk_tertile_curves is responsible for perform tertile curves.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def risk_tertile_curves(risk: np.ndarray, df_test: pd.DataFrame) -> dict:
    """Fit Kaplan-Meier curves for low, medium, and high predicted risk.

    Tertile boundaries are calculated from the supplied cohort's risk scores.
    The curves are descriptive stratification plots, not additional model
    fitting or threshold selection.
    """
    # Lower risk corresponds to a larger expected survival time, so the labels
    # are ordered from the smallest predicted risk to the largest.
    q33, q67 = np.percentile(risk, [33.3, 66.7])
    labels = np.where(risk <= q33, "Low", np.where(risk <= q67, "Medium", "High"))
    curves = {}
    for grp in ["Low", "Medium", "High"]:
        mask = labels == grp
        km = KaplanMeierFitter()
        km.fit(df_test[TIME_COL][mask], df_test[EVENT_COL][mask], label=grp)
        curves[grp] = km
    return curves


# ── 7. Calibration at fixed time points ──────────────────────────────────────

# Function guide: calibration_at is responsible for perform at.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def calibration_at(surv_mat: np.ndarray, df_test: pd.DataFrame,
                   times: list, n_bins: int = 5) -> dict:
    """
    O/E calibration at each time: bin predicted S(t) into n_bins groups,
    compare predicted mean vs KM-estimated observed survival.
    """
    # For each horizon, group patients by predicted survival probability and
    # compare the group mean prediction with its Kaplan-Meier estimate.
    results = {}
    for i, t in enumerate(times):
        pred = surv_mat[:, i]
        # Repeated predictions can collapse quantile bins; skip a horizon when
        # fewer than two usable groups remain after de-duplication.
        bins = np.percentile(pred, np.linspace(0, 100, n_bins + 1))
        bins = np.unique(bins)
        if len(bins) < 3:
            continue
        labels = np.digitize(pred, bins[1:-1])
        obs_pts, pred_pts = [], []
        for b in np.unique(labels):
            mask = labels == b
            km = KaplanMeierFitter()
            km.fit(df_test[TIME_COL][mask], df_test[EVENT_COL][mask])
            obs_pts.append(float(km.predict(t)))
            pred_pts.append(float(pred[mask].mean()))
        results[t] = {"observed": obs_pts, "predicted": pred_pts}
    return results


# ── 8. Persistence ───────────────────────────────────────────────────────────

# Function guide: env_versions is responsible for perform versions.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def env_versions() -> dict:
    """Return library versions needed to interpret a saved model artifact."""
    import optuna

    return dict(
        python   = platform.python_version(),
        numpy    = np.__version__,
        pandas   = pd.__version__,
        sklearn  = sklearn.__version__,
        lifelines= lifelines.__version__,
        sksurv   = sksurv.__version__,
        optuna   = optuna.__version__,
        torch    = (torch.__version__ if _DEEPSURV_AVAILABLE else "n/a"),
    )


# Function guide: save_model is responsible for save model.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def save_model(obj, path: str, meta: dict | None = None) -> None:
    """Serialize a model plus metadata and environment versions with joblib."""
    # Version metadata makes later bundle audits reproducible and helps explain
    # incompatibilities caused by changes in statistical libraries.
    bundle = dict(model=obj, meta=meta or {}, versions=env_versions(),
                  saved_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    joblib.dump(bundle, path, compress=3)
    print(f"  saved → {path}")


# Function guide: load_model is responsible for load model.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def load_model(path: str) -> dict:
    """Load the dictionary produced by :func:`save_model`."""
    return joblib.load(path)


# ── Quick self-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("survival_core self-test …")
    df = generate_nsclc_cohort(n=200, seed=SEED)
    print(f"  cohort: {len(df)} patients  "
          f"event_rate={df[EVENT_COL].mean():.1%}  "
          f"median_time={df[TIME_COL].median():.1f} months")

    dist_res = test_distributions(df)
    print(f"  best distribution: {dist_res['best']}  "
          f"(AIC={dist_res['table'].iloc[0]['AIC']:.1f})")

    cph = build_cox(df, penalizer=0.1)
    print(f"  Cox C-index (train): {cph.concordance_index_:.3f}")
    print("self-test OK")
