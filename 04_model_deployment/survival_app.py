"""
survival_app.py — TCGA-LUAD Lung Adenocarcinoma Overall Survival
================================================================
Cox Proportional Hazards  ×  Log-Logistic Accelerated Failure Time
Clinical data: TCGA-LUAD · N = 509 patients · NCI GDC API
Cell Press visual style · Research & educational use only
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
import threading
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
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


_BUNDLE_VERSION = 3


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
    times_arr = np.array(times, dtype=float)
    if len(times_arr) == 0:
        return {"times": [], "values": []}
    try:
        km_ref = KaplanMeierFitter().fit(df_train[TIME_COL], df_train[EVENT_COL])
        ref_sf = np.array([km_ref.predict(t) for t in times_arr], dtype=float)
        null_m = np.tile(ref_sf, (n_test, 1))
        _, null_bs = _brier_score(y_train, sc.administrative_censor(y_test, times_arr), null_m, times_arr)
        return {"times": times_arr.tolist(), "values": np.asarray(null_bs).tolist()}
    except Exception:
        return {"times": [], "values": []}


# Function guide: _sanitized_bundle is responsible for load, validate, or save deployment bundle state.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _sanitized_bundle(raw: dict) -> tuple[dict, bool]:
    df_train = raw.get("df_train")
    df_all = raw.get("df_all")
    y_train = raw["y_train"]
    y_test = raw["y_test"]
    n_train = int(raw.get("n_train", len(y_train)))
    n_test = int(raw.get("n_test", len(y_test)))
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

    keep = [
        "y_train", "y_test", "n_total", "ev_rate", "med_fu",
        "dist", "km_train", "cox", "aft", "res_cox", "res_aft",
        "tr_cox", "tr_aft", "imp_statistics", "feat_cols",
        "preprocessing_decisions", "provenance", "diagnostics",
        "preprocessor",
    ]
    clean = {k: raw[k] for k in keep if k in raw}
    clean.update(
        bundle_version=_BUNDLE_VERSION,
        n_train=n_train,
        n_test=n_test,
        stage_counts=stage_counts or {str(i): 0 for i in range(1, 5)},
        null_brier=null_brier or {"times": [], "values": []},
    )
    changed = set(raw) != set(clean) or raw.get("bundle_version") != _BUNDLE_VERSION
    # Remove fit-only caches. Prediction requires parameters, baselines, and
    # feature schema; no individual training feature matrix is needed.
    for name in ("cox", "aft"):
        model = clean[name]
        owners = [model, vars(model).get("_model")]
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
    hits      = _download_gdc()
    df        = _clean_clinical_records(hits)
    n_total   = len(df)
    ev_rate   = df[EVENT_COL].mean()
    med_fu    = df[TIME_COL].median()
    print(f"  N={n_total}  events={ev_rate:.0%}  median FU={med_fu:.0f}m", flush=True)

    df_train, df_test = train_test_split(
        df, test_size=0.20, random_state=SEED, stratify=df[EVENT_COL],
    )
    df_train = df_train.reset_index(drop=True)
    df_test  = df_test.reset_index(drop=True)

    ft_train, imp = _transform_feature_matrix(df_train, fit=True)
    ft_test,  _   = _transform_feature_matrix(df_test,  imputer=imp)

    y_train = _build_survival_target(df_train)
    y_test  = _build_survival_target(df_test)

    print("  Distribution fitting ...", flush=True)
    dist     = test_distributions(df_train)
    km_train = KaplanMeierFitter().fit(
        df_train[TIME_COL], df_train[EVENT_COL], label="Kaplan-Meier"
    )

    print("  Training Cox PH ...", flush=True)
    cox = build_cox(df_train, feat_df=ft_train, raw_feat_df=df_train[FEAT_COLS])
    print(f"    penalizer={cox._penalizer_used}  train C={cox.concordance_index_:.3f}",
          flush=True)

    print(f"  Training {dist['best']} AFT ...", flush=True)
    aft = build_aft(df_train, aft_class=dist["aft_class"], feat_df=ft_train)

    eval_times = sc._clip_times(EVAL_TIMES, y_train, y_test)
    print("  Evaluating test set (n_boot=150) ...", flush=True)
    res_cox = evaluate(cox, "cox", df_test, y_train, y_test,
                       eval_times, feat_df_test=ft_test, n_boot=150)
    res_aft = evaluate(aft, "aft", df_test, y_train, y_test,
                       eval_times, feat_df_test=ft_test, n_boot=150)
    print(
        f"  Cox  C={res_cox['c_index']:.3f} [{res_cox['ci_lo']:.3f},{res_cox['ci_hi']:.3f}]"
        f"  IBS={res_cox['ibs']:.4f}", flush=True,
    )
    print(
        f"  AFT  C={res_aft['c_index']:.3f} [{res_aft['ci_lo']:.3f},{res_aft['ci_hi']:.3f}]"
        f"  IBS={res_aft['ibs']:.4f}", flush=True,
    )

    print("  Computing training-set metrics ...", flush=True)
    tr_cox = sc.evaluate_train(cox, "cox", df_train, y_train,
                               eval_times, feat_df_train=ft_train)
    tr_aft = sc.evaluate_train(aft, "aft", df_train, y_train,
                               eval_times, feat_df_train=ft_train)
    print(f"  Cox  train C={tr_cox['c_index']:.3f}  train IBS={tr_cox['ibs']:.4f}",
          flush=True)
    print(f"  AFT  train C={tr_aft['c_index']:.3f}  train IBS={tr_aft['ibs']:.4f}",
          flush=True)

    null_brier = _null_brier_curve(df_train, y_train, y_test, len(df_test),
                                   res_cox.get("times", EVAL_TIMES))

    for label, result in (("Cox test", res_cox), ("AFT test", res_aft), ("Cox train", tr_cox), ("AFT train", tr_aft)):
        if not all(np.isfinite(result[key]) for key in ("c_index", "ibs")) or result.get("metric_errors"):
            raise RuntimeError(f"Invalid {label} evaluation: {result.get('metric_errors', {})}")

    decisions, diagnostics = sc.audit_clinical_preprocessing(df, df_train, ft_train, cox, imp)
    provenance = {
        "seed": SEED, "split": "80/20 event-stratified", "n_train": len(df_train), "n_test": len(df_test),
        "train_events": int(df_train[EVENT_COL].sum()), "test_events": int(df_test[EVENT_COL].sum()),
        "cleaning": df.attrs.get("cleaning_audit", {}),
        "source_sha256": hashlib.sha256(json.dumps(hits, sort_keys=True).encode()).hexdigest(),
        "cox_cv": "Five event-stratified folds; imputation, functional screening, spline knots and dummy levels refitted in each training fold",
        "evaluation_times": eval_times.tolist(),
        "versions": {"lifelines": sc.lifelines.__version__, "scikit_survival": sc.sksurv.__version__, "scikit_learn": sc.sklearn.__version__},
    }
    audit_dir = Path(os.environ.get(
        "SURVIVAL_AUDIT_DIR",
        str(Path(__file__).parent / ".cache" / "audit"),
    ))
    audit_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(audit_dir / "tcga_luad_cleaned.csv", index=False)
    ft_train.to_csv(audit_dir / "X_train_design.csv", index=False)
    ft_test.to_csv(audit_dir / "X_test_design.csv", index=False)
    (audit_dir / "data_dictionary.json").write_text(json.dumps(decisions, indent=2), encoding="utf-8")
    (Path(__file__).parent / "preprocessing_decisions.json").write_text(
        json.dumps({"variables": decisions, "provenance": provenance, "diagnostics": diagnostics}, indent=2, allow_nan=False), encoding="utf-8",
    )
    bundle = dict(
        bundle_version=_BUNDLE_VERSION,
        y_train=y_train, y_test=y_test,
        n_total=n_total, ev_rate=ev_rate, med_fu=med_fu,
        n_train=len(df_train), n_test=len(df_test),
        stage_counts=_stage_counts(df),
        dist=dist, km_train=km_train,
        cox=cox, aft=aft,
        res_cox=res_cox, res_aft=res_aft,
        tr_cox=tr_cox, tr_aft=tr_aft,
        null_brier=null_brier,
        imp_statistics=imp.statistics_.tolist(), feat_cols=FEAT_COLS,
        preprocessing_decisions=decisions, provenance=provenance, diagnostics=diagnostics,
        preprocessor=imp,
    )
    return _sanitized_bundle(bundle)[0]


print("=" * 56, flush=True)
print("TCGA-LUAD Survival App — initialising", flush=True)
print("=" * 56, flush=True)

if _BUNDLE_PATH.exists() and os.environ.get("SURVIVAL_REBUILD_BUNDLE") != "1":
    _raw_bundle = _startup_from_bundle(_BUNDLE_PATH)
    if _raw_bundle.get("bundle_version") != _BUNDLE_VERSION:
        raise RuntimeError("Outdated survival bundle. Run 03_model_training_and_evaluation/train_and_export_model_bundle.py before starting the app.")
    _B, _changed = _sanitized_bundle(_raw_bundle)
    if _changed:
        _save_bundle(_B, _BUNDLE_PATH)
        print(f"  Bundle sanitized: {_BUNDLE_PATH}", flush=True)
else:
    _B = _startup_from_scratch()
    _save_bundle(_B, _BUNDLE_PATH)
    print(f"  Bundle saved: {_BUNDLE_PATH}", flush=True)

# ── Unpack bundle ─────────────────────────────────────────────────────────────
# The fitted transformer stores training statistics, spline knots and categories.
_IMP = _B["preprocessor"]

N_TOTAL  = _B["n_total"]
EV_RATE  = _B["ev_rate"]
MED_FU   = _B["med_fu"]
N_TRAIN  = _B["n_train"]
N_TEST   = _B["n_test"]
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

_best_name = ("Cox PH" if RES_COX["c_index"] >= RES_AFT["c_index"]
              else f"{DIST['best']} AFT")
print("  Ready.", flush=True)


# ── 3. Analytics: global visitor map ─────────────────────────────────────────
# Runtime integrations are optional. The app remains functional when these
# variables are absent, with visit logging and geo-enrichment disabled.
_IPINFO_TOKEN = os.environ.get("IPINFO_TOKEN", "").strip()
_SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
_SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "").strip()
_ANALYTICS_CONFIGURED = bool(
    _SUPABASE_URL and _SUPABASE_KEY
)
_ANALYTICS_STATE = {
    "error": None,
    "mode": "remote" if _ANALYTICS_CONFIGURED else "local",
    "retry_at": 0.0,
}
_APP_NAME_SV  = "nsclc-survival"
_ANALYTICS_RETRY_SECONDS = 60.0
_LOCAL_VISITS_LIMIT_SV = 500
_LOCAL_VISITS_SV: list[dict] = []
_LOCAL_VISITS_LOCK_SV = threading.Lock()

_COUNTRY_NAMES = {
    "AF":"Afghanistan","AL":"Albania","DZ":"Algeria","AR":"Argentina",
    "AU":"Australia","AT":"Austria","BD":"Bangladesh","BE":"Belgium",
    "BR":"Brazil","BG":"Bulgaria","CA":"Canada","CL":"Chile",
    "CN":"China","CO":"Colombia","HR":"Croatia","CZ":"Czech Republic",
    "DK":"Denmark","EG":"Egypt","FI":"Finland","FR":"France",
    "DE":"Germany","GH":"Ghana","GR":"Greece","HK":"Hong Kong",
    "HU":"Hungary","IN":"India","ID":"Indonesia","IR":"Iran",
    "IQ":"Iraq","IE":"Ireland","IL":"Israel","IT":"Italy",
    "JP":"Japan","JO":"Jordan","KZ":"Kazakhstan","KE":"Kenya",
    "KR":"South Korea","KW":"Kuwait","LB":"Lebanon","MY":"Malaysia",
    "MX":"Mexico","MA":"Morocco","NL":"Netherlands","NZ":"New Zealand",
    "NG":"Nigeria","NO":"Norway","PK":"Pakistan","PE":"Peru",
    "PH":"Philippines","PL":"Poland","PT":"Portugal","QA":"Qatar",
    "RO":"Romania","RU":"Russia","SA":"Saudi Arabia","SG":"Singapore",
    "ZA":"South Africa","ES":"Spain","SE":"Sweden","CH":"Switzerland",
    "TW":"Taiwan","TH":"Thailand","TN":"Tunisia","TR":"Turkey",
    "UA":"Ukraine","AE":"United Arab Emirates","GB":"United Kingdom",
    "US":"United States","VN":"Vietnam","YE":"Yemen","ZW":"Zimbabwe",
}


# Function guide: _country_name is responsible for perform the routine-specific application step.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _country_name(code: str) -> str:
    return _COUNTRY_NAMES.get((code or "").upper(), code or "")


# Function guide: _lookup_ip_location is responsible for perform the routine-specific application step.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _lookup_ip_location(ip: str):
    if not ip or ip in ("127.0.0.1", "::1"):
        return None, None, None, None
    try:
        if _IPINFO_TOKEN:
            response = requests.get(
                f"https://ipinfo.io/{ip}/json",
                params={"token": _IPINFO_TOKEN},
                timeout=4,
            )
            if response.status_code != 200:
                return None, None, None, None
            data = response.json()
            loc = data.get("loc", "")
            lat, lon = map(float, loc.split(",")) if loc else (None, None)
            return data.get("country"), data.get("city"), lat, lon

        response = requests.get(f"https://ipwho.is/{ip}", timeout=4)
        data = response.json() if response.status_code == 200 else {}
        if not data.get("success"):
            return None, None, None, None
        return (
            data.get("country_code"),
            data.get("city"),
            data.get("latitude"),
            data.get("longitude"),
        )
    except Exception:
        return None, None, None, None


# Function guide: _sb_headers_sv is responsible for perform the routine-specific application step.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _sb_headers_sv():
    return {
        "apikey":        _SUPABASE_KEY,
        "Authorization": f"Bearer {_SUPABASE_KEY}",
        "Content-Type":  "application/json",
        "Prefer":        "return=minimal",
    }


# One shared session per process: connection pooling + shorter TLS handshakes
# for the (otherwise) many small analytics requests.
_HTTP_SESSION_SV = requests.Session()
_HTTP_SESSION_SV.headers.update({"User-Agent": _APP_NAME_SV})


# Function guide: _map_coordinate_sv is responsible for perform the routine-specific application step.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _map_coordinate_sv(value, lower, upper):
    """Return a finite coordinate inside the requested range."""
    try:
        coordinate = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(coordinate) or not lower <= coordinate <= upper:
        return None
    return coordinate


# Function guide: _normalise_visit_sv is responsible for perform the routine-specific application step.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _normalise_visit_sv(visit):
    if not isinstance(visit, dict):
        return None
    lat = _map_coordinate_sv(visit.get("lat"), -90.0, 90.0)
    lon = _map_coordinate_sv(visit.get("lon"), -180.0, 180.0)
    if lat is None or lon is None:
        return None
    return {
        "country": str(visit.get("country") or "").strip(),
        "city": str(visit.get("city") or "").strip(),
        "lat": lat,
        "lon": lon,
    }


# Function guide: _normalise_visits_sv is responsible for perform the routine-specific application step.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _normalise_visits_sv(visits):
    normalised = (_normalise_visit_sv(visit) for visit in (visits or []))
    return [visit for visit in normalised if visit is not None]


# Function guide: _record_local_visit_sv is responsible for perform the routine-specific application step.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _record_local_visit_sv(country, city, lat, lon):
    visit = _normalise_visit_sv({
        "country": country,
        "city": city,
        "lat": lat,
        "lon": lon,
    })
    if visit is None:
        return
    with _LOCAL_VISITS_LOCK_SV:
        _LOCAL_VISITS_SV.append(visit)
        del _LOCAL_VISITS_SV[:-_LOCAL_VISITS_LIMIT_SV]


# Function guide: _local_visits_sv is responsible for perform the routine-specific application step.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _local_visits_sv():
    with _LOCAL_VISITS_LOCK_SV:
        return [dict(visit) for visit in _LOCAL_VISITS_SV]


# Function guide: _log_visit_sv is responsible for perform the routine-specific application step.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _log_visit_sv(country, city, lat, lon):
    if not (_SUPABASE_URL and _SUPABASE_KEY):
        return
    if lat is None or lon is None:
        return
    try:
        _HTTP_SESSION_SV.post(
            f"{_SUPABASE_URL}/rest/v1/visits",
            headers=_sb_headers_sv(),
            json={"app_name": _APP_NAME_SV, "country": country,
                  "city": city, "lat": lat, "lon": lon},
            timeout=5,
        )
    except Exception:
        pass


# Function guide: _fetch_visits_sv is responsible for perform the routine-specific application step.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _fetch_visits_sv():
    if not (_SUPABASE_URL and _SUPABASE_KEY):
        _ANALYTICS_STATE["error"] = "configuration"
        _ANALYTICS_STATE["mode"] = "local"
        return _local_visits_sv()
    if time.monotonic() < _ANALYTICS_STATE["retry_at"]:
        return _local_visits_sv()
    try:
        hdrs = {k: v for k, v in _sb_headers_sv().items() if k != "Prefer"}
        r = _HTTP_SESSION_SV.get(
            f"{_SUPABASE_URL}/rest/v1/visits",
            headers=hdrs,
            params={"app_name": f"eq.{_APP_NAME_SV}",
                    "select": "country,city,lat,lon"},
            timeout=5,
        )
        if r.status_code == 200:
            data = r.json()
            _ANALYTICS_STATE["error"] = None
            _ANALYTICS_STATE["mode"] = "remote"
            _ANALYTICS_STATE["retry_at"] = 0.0
            return _normalise_visits_sv(data if isinstance(data, list) else [])
        _ANALYTICS_STATE["error"] = f"http-{r.status_code}"
        _ANALYTICS_STATE["mode"] = "local"
        _ANALYTICS_STATE["retry_at"] = (
            time.monotonic() + _ANALYTICS_RETRY_SECONDS
        )
        return _local_visits_sv()
    except Exception:
        _ANALYTICS_STATE["error"] = "connection"
        _ANALYTICS_STATE["mode"] = "local"
        _ANALYTICS_STATE["retry_at"] = (
            time.monotonic() + _ANALYTICS_RETRY_SECONDS
        )
        return _local_visits_sv()


_WORLD_GEO_PATH_SV = Path(__file__).parent / "world.geojson"
_WORLD_GEO_SV = None
_WORLD_PATCHES_SV = None


# Function guide: _load_world_geo_sv is responsible for perform the routine-specific application step.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _load_world_geo_sv():
    global _WORLD_GEO_SV
    if _WORLD_GEO_SV is None and _WORLD_GEO_PATH_SV.exists():
        with open(_WORLD_GEO_PATH_SV, encoding="utf-8") as _f:
            _WORLD_GEO_SV = json.load(_f)
    return _WORLD_GEO_SV


# Function guide: _world_patches_sv is responsible for perform the routine-specific application step.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _world_patches_sv():
    """World-outline matplotlib Polygons, parsed once per process."""
    global _WORLD_PATCHES_SV
    if _WORLD_PATCHES_SV is not None:
        return _WORLD_PATCHES_SV
    from matplotlib.patches import Polygon

    geo = _load_world_geo_sv()
    patches = []
    if geo:
        for feat in geo.get("features", []):
            geom = feat.get("geometry") or {}
            gtype = geom.get("type", "")
            coords = geom.get("coordinates", [])
            try:
                if gtype == "Polygon":
                    pts = np.array(coords[0])[:, :2]
                    patches.append(Polygon(pts, closed=True))
                elif gtype == "MultiPolygon":
                    for poly in coords:
                        pts = np.array(poly[0])[:, :2]
                        patches.append(Polygon(pts, closed=True))
            except Exception:
                pass
    _WORLD_PATCHES_SV = patches
    return patches


# Function guide: _make_visit_map_sv is responsible for create a user-facing visualization or UI component.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _make_visit_map_sv(visits, user_lat=None, user_lon=None,
                       analytics_mode="remote"):
    from matplotlib.collections import PatchCollection

    valid = _normalise_visits_sv(visits)
    point_groups = {}
    for visit in valid:
        key = (
            visit["city"], visit["country"],
            round(visit["lat"], 2), round(visit["lon"], 2),
        )
        point_groups[key] = point_groups.get(key, 0) + 1
    points = sorted(
        ((*key, count) for key, count in point_groups.items()),
        key=lambda item: (-item[4], item[0], item[1]),
    )

    user_lat = _map_coordinate_sv(user_lat, -90.0, 90.0)
    user_lon = _map_coordinate_sv(user_lon, -180.0, 180.0)

    fig, ax = plt.subplots(figsize=(7.0, 3.5), facecolor="white")
    ax.set_facecolor("white")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-70, 85)
    _cell_ax(fig, ax)

    patches = _world_patches_sv()
    if patches:
        pc = PatchCollection(
            patches, facecolor=BRAND["mint"], edgecolor=BRAND["edge"],
            linewidth=0.4, alpha=0.35, zorder=1,
        )
        ax.add_collection(pc)

    if points:
        lons = [point[3] for point in points]
        lats = [point[2] for point in points]
        sizes = [min(80.0, 22.0 + 12.0 * np.sqrt(point[4]))
                 for point in points]
        ax.scatter(lons, lats, s=sizes, color=BRAND["blue"], alpha=0.85,
                   marker="o", zorder=3, linewidths=0,
                   label=f"Visitors (n={len(valid)})")
        labelled_points = [point for point in points if point[0]][:4]
        for i, (name, _country, lat_c, lon_c, count) in enumerate(labelled_points):
            if lon_c < -100:
                dx = 5
            elif lon_c > 105:
                dx = -4
            else:
                dx = 5 if i % 2 == 0 else -4
            dy = 4 if i % 3 == 0 else (-7 if i % 3 == 1 else 9)
            label = f"{name} ({count})" if count > 1 else name
            ax.annotate(label, xy=(lon_c, lat_c),
                        xytext=(dx, dy), textcoords="offset points",
                        ha="left" if dx >= 0 else "right",
                        fontsize=7.0, color=BRAND["ink"], zorder=5, clip_on=False)

    if user_lat is not None and user_lon is not None:
        ax.scatter([user_lon], [user_lat], s=32, color=BRAND["red"],
                   marker="o", zorder=4, linewidths=0, label="You")

    if not points and user_lat is None:
        empty_label = (
            "Waiting for a live visitor"
            if analytics_mode == "local" else "No mapped visits yet"
        )
        ax.text(
            0.5, 0.055, empty_label,
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=7.5, color=_MUTED, zorder=5,
        )

    ax.set_xticks([])
    ax.set_yticks([])

    if points or user_lat is not None:
        ax.legend(fontsize=7.5, loc="lower left", frameon=False)

    fig.tight_layout(pad=0.6)
    return fig

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
    """Survival probabilities at _CURVE_T for one patient."""
    try:
        sf    = model.predict_survival_function(feat_df, times=_CURVE_T)
        probs = sf.iloc[:, 0].values
    except TypeError:
        sf    = model.predict_survival_function(feat_df)
        s     = sf.iloc[:, 0]
        probs = np.interp(_CURVE_T, s.index.values, s.values, left=1.0)
    return probs.clip(0.0, 1.0)


# Function guide: _survival_at_times is responsible for prepare or evaluate survival model information.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _survival_at_times(model, times: np.ndarray) -> np.ndarray:
    vals = model.survival_function_at_times(times)
    arr = np.asarray(vals, dtype=float).reshape(-1)
    return np.clip(arr, 0.0, 1.0)


# Function guide: _survival_function_frame is responsible for prepare or evaluate survival model information.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _survival_function_frame(model):
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

# Keep the active interface theme in one deployable stylesheet.
_CSS = (Path(__file__).parent / "compact_theme.css").read_text(encoding="utf-8")

_TEXT_SCALE_JS = """
(() => {
  const storageKey = "medictio-text-scale";
  const allowed = new Set(["1", "1.125", "1.25"]);
  const applyScale = (value) => {
    const scale = allowed.has(value) ? value : "1.125";
    document.documentElement.style.setProperty("--app-scale", scale);
    return scale;
  };
  document.addEventListener("DOMContentLoaded", () => {
    const control = document.getElementById("text-scale");
    if (!control) return;
    let saved = "1.125";
    try { saved = localStorage.getItem(storageKey) || saved; } catch (_) {}
    control.value = applyScale(saved);
    control.addEventListener("change", () => {
      const value = applyScale(control.value);
      try { localStorage.setItem(storageKey, value); } catch (_) {}
    });
  });
})();
"""


_stage_lbl  = {"1": "I",  "2": "II",  "3": "III",  "4": "IV"}
_t_lbl      = {"1": "T1", "2": "T2",  "3": "T3",   "4": "T4"}
_n_lbl      = {"0": "N0", "1": "N1",  "2": "N2",   "3": "N3"}
_m_lbl      = {"0": "M0 — No distant metastasis", "1": "M1 — Distant metastasis"}


# Function guide: _overview_tile is responsible for perform the routine-specific application step.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _overview_tile(label: str, value: str, detail: str, accent: str) -> ui.Tag:
    return ui.tags.div(
        ui.tags.div(label, class_="overview-label"),
        ui.tags.div(value, class_="overview-value"),
        ui.tags.div(detail, class_="overview-detail"),
        class_=f"overview-tile {accent}",
    )


# Function guide: _section_head is responsible for perform the routine-specific application step.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _section_head(_kicker: str, title: str, copy: str) -> ui.Tag:
    return ui.tags.div(
        ui.tags.h4(title, class_="section-title"),
        ui.tags.p(copy, class_="section-copy"),
        class_="section-head",
    )


# Function guide: _hero_metadata is responsible for perform the routine-specific application step.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _hero_metadata() -> ui.Tag:
    return ui.tags.div(
        ui.tags.span(ui.tags.strong(f"{N_TOTAL}"), " patients", class_="hero-meta-item"),
        ui.tags.span(
            ui.tags.strong(f"{N_TRAIN} / {N_TEST}"),
            " train / test",
            class_="hero-meta-item",
        ),
        ui.tags.span(
            ui.tags.strong("80 / 20"),
            f" event-stratified, seed {SEED}",
            class_="hero-meta-item",
        ),
        ui.tags.span(ui.tags.strong(f"{EV_RATE:.0%}"), " observed events", class_="hero-meta-item"),
        ui.tags.span(
            ui.tags.strong(f"{len(FEAT_COLS)}"),
            " inputs: age, stage, T, N, M",
            class_="hero-meta-item",
        ),
        ui.tags.span(
            "Training-only preprocessing; held-out test evaluation.",
            class_="hero-meta-note",
        ),
        class_="hero-meta",
        **{"aria-label": "Dataset and split overview"},
    )


# Function guide: _stage_tile is responsible for perform the routine-specific application step.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _stage_tile(stage_code: str) -> ui.Tag:
    count = int(STAGE_COUNTS.get(stage_code, 0))
    share = (100.0 * count / N_TOTAL) if N_TOTAL else 0.0
    return ui.tags.div(
        ui.tags.div(f"Stage {_stage_lbl[stage_code]}", class_="stage-kicker"),
        ui.tags.div(f"{count}", class_="stage-value"),
        ui.tags.div(f"{share:.0f}% of cohort", class_="stage-detail"),
        class_=f"stage-tile stage-{stage_code}",
    )


# Function guide: _note_block is responsible for perform the routine-specific application step.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _note_block(title: str, copy: str) -> ui.Tag:
    return ui.tags.div(
        ui.tags.div(title, class_="note-title"),
        ui.tags.p(copy, class_="note-copy"),
        class_="note-block",
    )

# Function guide: _input_panel is responsible for perform the routine-specific application step.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _input_panel():
    return ui.tags.aside(
        ui.tags.h4("Patient profile", class_="input-title"),
        ui.tags.p("Enter age and pathologic stage.",
                  class_="input-copy"),
        ui.tags.div("Patient & overall stage", class_="input-group-label"),
        ui.tags.div(
            ui.input_numeric("age", "Age (years)", value=65, min=18, max=100, step=1),
            ui.input_select("stage", "AJCC stage",
                            {k: f"Stage {v}" for k, v in _stage_lbl.items()}, selected="2"),
            class_="input-grid",
        ),
        ui.tags.div("Pathologic TNM categories", class_="input-group-label"),
        ui.tags.div(
            ui.input_select("t_stage", "Tumor (T)", _t_lbl, selected="2"),
            ui.input_select("n_stage", "Nodes (N)", _n_lbl, selected="0"),
            class_="input-grid",
        ),
        ui.input_select("m_stage", "Metastasis (M)", _m_lbl, selected="0"),
        ui.input_action_button("submit", "Update Prediction", class_="btn btn-primary w-100"),
        ui.tags.p("Inputs are applied when you update the prediction.", class_="input-hint"),
        class_="input-panel", **{"aria-label": "Patient inputs"},
    )


# Function guide: _view_notes is responsible for perform the routine-specific application step.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def _view_notes(*notes):
    return ui.tags.details(
        ui.tags.summary("Reading guide"),
        ui.tags.div(*notes, class_="note-grid"),
        class_="detail-panel view-notes",
        open=True,
    )


app_ui = ui.page_fluid(
    ui.tags.style(_CSS),
    ui.tags.script(_TEXT_SCALE_JS),
    ui.tags.div(
        ui.tags.div(
            ui.tags.h1("Lung Cancer Survival", class_="page-title"),
            ui.tags.p(
                f"TCGA-LUAD | Cox PH and {DIST['best']} AFT models",
                class_="page-subtitle",
            ),
            _hero_metadata(),
            class_="hero-copy",
        ),
        ui.tags.div(
            ui.tags.span("Research use only", class_="app-status"),
            ui.tags.label("Text size", class_="text-scale-label", **{"for": "text-scale"}),
            ui.tags.select(
                ui.tags.option("Standard", value="1"),
                ui.tags.option("Comfortable", value="1.125", selected="selected"),
                ui.tags.option("Large", value="1.25"),
                id="text-scale",
                class_="text-scale-select",
                **{"aria-label": "Interface text size"},
            ),
            class_="hero-actions",
        ),
        class_="hero-banner",
    ),

    ui.navset_tab(
        ui.nav_panel(
            "Patient Prediction",
            _section_head(
                "Individual prediction",
                "Patient prediction",
                "Enter a patient profile to compare overall survival estimates from two models.",
            ),
            ui.tags.div(
                _input_panel(),
                ui.tags.div(
                    ui.card(
                        ui.tags.div(ui.output_ui("survival_curve"),
                                    class_="chart-image chart-wide desktop-chart"),
                        ui.tags.div(ui.output_ui("survival_curve_mobile"),
                                    class_="chart-image chart-square mobile-chart"),
                    ),
                    ui.tags.details(
                        ui.tags.summary("All time points and model differences"),
                        ui.output_ui("prob_table"), class_="detail-panel",
                    ),
                    class_="prediction-results",
                ),
                ui.tags.div(ui.output_ui("info_bar"), class_="prediction-milestones"),
                class_="prediction-workspace",
            ),
            _view_notes(
                _note_block(
                    "Inputs used",
                    "Age, AJCC pathologic stage, and pathologic T, N, and M categories are used as predictors in both models. Missing age uses the training median; missing stage categories use the training mode.",
                ),
                _note_block(
                    "Model pairing",
                    f"Cox PH and {DIST['best']} AFT estimates are shown together so agreement and divergence between the models remain visible.",
                ),
                _note_block(
                    "Intended use",
                    "This interface is intended for research and education. It displays training and test performance and is not a clinical decision-support system.",
                ),
            ),
            ui.tags.p(
                "Based on TCGA-LUAD data. For research and educational use only; "
                "not intended for clinical decision-making or individual prognostic counseling.",
                class_="disclaimer",
            ),
        ),

        ui.nav_panel(
            "Cohort Overview",
            _section_head(
                "Cohort overview",
                "Stage distribution and marginal survival",
                "Review stage composition, compare candidate marginal survival distributions, and view aggregate visitor activity.",
            ),
            ui.tags.div(
                _overview_tile(
                    "Cohort",
                    f"{N_TOTAL}",
                    f"{N_TRAIN} training / {N_TEST} test patients",
                    "accent-blue",
                ),
                _overview_tile(
                    "Observed events",
                    f"{EV_RATE:.0%}",
                    "overall survival event rate in the full cohort",
                    "accent-teal",
                ),
                _overview_tile(
                    "Follow-up",
                    f"{MED_FU:.0f} m",
                    "median observed follow-up time",
                    "accent-navy",
                ),
                class_="overview-grid cohort-overview",
            ),
            ui.tags.div(
                _stage_tile("1"),
                _stage_tile("2"),
                _stage_tile("3"),
                _stage_tile("4"),
                class_="stage-grid",
            ),
            ui.layout_columns(
                ui.card(
                    ui.tags.div(
                        ui.output_ui("dist_plot"),
                        class_="chart-image chart-square",
                    ),
                    class_="equal-card",
                ),
                ui.card(
                    ui.card_header("AIC / BIC Comparison"),
                    ui.tags.div(
                        ui.output_ui("dist_table"),
                        ui.tags.div(
                            ui.tags.h4(
                                "Why parametric distribution selection matters",
                                style=f"font-size:.64rem;font-weight:700;color:{_MUTED};"
                                      "text-transform:uppercase;letter-spacing:.6px;"
                                      "margin:12px 0 5px;",
                            ),
                            ui.tags.p(
                                "The Akaike Information Criterion (AIC) balances goodness "
                                "of fit against model complexity. Among the candidate "
                                "families, the distribution with the lowest AIC provides "
                                "the preferred marginal fit.",
                                style="font-size:.70rem;line-height:1.48;color:var(--ct);",
                            ),
                            ui.tags.p(
                                "Best fit: ",
                                ui.tags.strong(DIST["best"]),
                                " — selected as the AFT family for the parametric model. "
                                "Unlike Cox PH, this parametric form permits model-based "
                                "extrapolation beyond the 72-month follow-up window; such "
                                "estimates should be interpreted cautiously.",
                                style="font-size:.70rem;line-height:1.48;color:var(--ct);",
                            ),
                            style="padding:2px 2px 0;",
                        ),
                        class_="result-frame result-dist",
                    ),
                    class_="equal-card",
                ),
                col_widths=[6, 6],
            ),
            ui.tags.details(
                ui.tags.summary("Visitor activity"),
                ui.layout_columns(
                ui.card(
                    ui.card_header("Global Visitor Map"),
                    ui.tags.div(
                        ui.output_ui("visit_map"),
                        class_="plot-frame plot-map",
                    ),
                    class_="equal-card",
                ),
                ui.card(
                    ui.card_header("Visit Statistics"),
                    ui.tags.div(ui.output_ui("visit_stats"), class_="result-frame result-map"),
                    class_="equal-card",
                ),
                col_widths=[8, 4],
                ),
                class_="detail-panel",
            ),
        ),

        ui.nav_panel(
            "Model Evaluation",
            _section_head(
                "Performance comparison",
                "Training and test performance",
                "Compare discrimination, prediction error, uncertainty, and training-to-test differences for both survival models.",
            ),
            ui.tags.p(
                f"Best held-out C-index: {max(RES_COX['c_index'], RES_AFT['c_index']):.3f} "
                f"({_best_name}). Training metrics represent apparent performance.",
                class_="evaluation-context",
            ),
            ui.output_ui("perf_chips"),
            ui.tags.div(
                ui.card(ui.tags.div(ui.output_ui("perf_plot_cindex"),
                                    class_="chart-image chart-square")),
                ui.card(ui.tags.div(ui.output_ui("perf_plot_auc"),
                                    class_="chart-image chart-square")),
                class_="chart-grid",
            ),
            _view_notes(
                _note_block(
                    "Bootstrap uncertainty",
                    "Test-set C-index intervals are estimated with 150 bootstrap resamples so rank differences are not read as exact.",
                ),
                _note_block(
                    "IBS horizon",
                    "Integrated Brier Score is evaluated at 12, 24, 36, 48, and 60 months after clipping to the observed support.",
                ),
                _note_block(
                    "Overfitting check",
                    "Training and test estimates are shown together. A large difference may indicate overfitting.",
                ),
            ),
        ),

        ui.nav_panel(
            "Methods",
            _section_head(
                "Reproducibility",
                "Model specification, preprocessing, and evaluation",
                "The information below is loaded from the deployed bundle so the model assumptions and evaluation procedures remain inspectable.",
            ),
            ui.output_ui("methods_panel"),
        ),

        id="main_tab",
    ),

    title="TCGA-LUAD Survival Dashboard",
    class_="dashboard-shell",
)


# ── 6. Server ────────────────────────────────────────────────────────────────

# Function guide: server is responsible for register the reactive Shiny server logic.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def server(input, output, session):

    # ── Visit logging ─────────────────────────────────────────────────────────
    _user_loc = {
        "country": "", "city": "", "lat": None, "lon": None,
    }
    try:
        _hdrs = session.http_conn.headers
        _ip = (
            _hdrs.get("x-forwarded-for") or
            _hdrs.get("x-real-ip") or ""
        ).split(",")[0].strip()
    except Exception:
        _ip = ""

    # Function guide: _do_log is responsible for perform the routine-specific application step.
    # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def _do_log(ip: str) -> None:
        country, city, lat, lon = _lookup_ip_location(ip)
        visit = _normalise_visit_sv({
            "country": country, "city": city, "lat": lat, "lon": lon,
        })
        if visit is not None:
            _user_loc.update(visit)
            _record_local_visit_sv(**visit)
        _log_visit_sv(country, city, lat, lon)

    threading.Thread(target=_do_log, args=(_ip,), daemon=True).start()

    # Auto-refresh the map/stats a few times after load so the visitor's own
    # just-logged visit (written asynchronously above) appears without a
    # manual page refresh. Bounded to a handful of ticks, then stops.
    _refresh_tick = reactive.value(0)
    _refresh_n    = {"c": 0}

    @reactive.effect
    # Function guide: _auto_refresh is responsible for perform the routine-specific application step.
    # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def _auto_refresh():
        _refresh_n["c"] += 1
        if _refresh_n["c"] <= 3:
            reactive.invalidate_later(3)
            _refresh_tick.set(_refresh_n["c"])

    @reactive.calc
    # Function guide: _visits is responsible for perform the routine-specific application step.
    # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def _visits():
        _refresh_tick.get()  # re-fetch when the tick advances
        return _fetch_visits_sv()

    @render.ui
    # Function guide: visit_map is responsible for perform the routine-specific application step.
    # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def visit_map():
        fig = _make_visit_map_sv(
            _visits(), _user_loc["lat"], _user_loc["lon"],
            _ANALYTICS_STATE["mode"],
        )
        fig.set_size_inches(7, 3.5)
        fig.axes[0].set_title("Global visitor activity", loc="left", fontsize=10, fontweight="bold")
        charts.finish(fig, "Source: app visit logs. Locations are approximate; unavailable locations are omitted.")
        return ui.tags.img(src=charts.png(fig), alt="Aggregate visitor locations on a world map")

    @render.ui
    # Function guide: visit_stats is responsible for perform the routine-specific application step.
    # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def visit_stats():
        from collections import Counter
        visits = _normalise_visits_sv(_visits())
        total  = len(visits)
        counts = Counter(
            f"{v.get('city')}, {_country_name(v.get('country'))}" if v.get("country") else v.get("city")
            for v in visits if v.get("city")
        )
        top    = counts.most_common(10)
        rows   = "".join(
            f"<tr><td style='font-size:.70rem;'>{i+1}. {html.escape(str(c))}</td>"
            f"<td class='num' style='font-size:.70rem;'>{n}</td></tr>"
            for i, (c, n) in enumerate(top)
        )
        empty_message = (
            "Waiting for the first mappable visitor."
            if not total else "No city labels are available yet."
        )
        empty = (
            f'<p style="font-size:.64rem;color:{_MUTED};padding:8px 0;">'
            f"{empty_message}</p>"
            if not top else ""
        )
        scope_note = ""
        if _ANALYTICS_STATE["mode"] == "local":
            scope_note = (
                f'<p style="font-size:.61rem;color:{_MUTED};margin:8px 0 0;'
                'line-height:1.35;">Live locations for this running app '
                'instance. Persistent history is currently unavailable.</p>'
            )
        return ui.HTML(f"""
<div style="padding:4px 6px;">
  <div style="text-align:center;margin-bottom:12px;">
    <div style="font-size:1.45rem;font-weight:700;color:{_COX_CLR};">{total}</div>
    <div style="font-size:.62rem;color:{_MUTED};text-transform:uppercase;
                letter-spacing:.8px;">Total Visits</div>
  </div>
  <table class="prob-tbl" style="width:100%;">
    <thead><tr>
      <th>City</th>
      <th style="text-align:right;">Visits</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table>
  {empty}
  {scope_note}
</div>
""")

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
        df_pt = pd.DataFrame([row])
        feat, _ = _transform_feature_matrix(df_pt, imputer=_IMP)
        return feat

    @reactive.calc
    # Function guide: curves is responsible for perform the routine-specific application step.
    # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def curves():
        feat   = patient_feat()
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
        for label, month in (("1-yr survival", 12), ("3-yr survival", 36), ("5-yr survival", 60)):
            first = float(np.interp(month, _CURVE_T, cox))
            second = float(np.interp(month, _CURVE_T, aft))
            chips.append(
                f'<div class="metric-chip" style="--tile:{_COX_CLR};--tint:#FFFFFF;">'
                f'<span class="mc-label">{label}</span>'
                f'<span class="model-value" style="color:{_COX_CLR};">'
                f'<small>Cox PH</small><strong>{first:.1%}</strong></span>'
                f'<span class="model-value" style="color:{_AFT_CLR};">'
                f'<small>AFT</small><strong>{second:.1%}</strong></span></div>'
            )
        return ui.HTML('<div class="infobar">' + "".join(chips) + "</div>")

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
            diff_str = (f'+{diff*100:.1f}%' if diff >= 0 else f'{diff*100:.1f}%')
            rows += (
                f'<tr>'
                f'<td>{int(t):d} m</td>'
                f'<td class="val cox">{sc*100:.1f}%</td>'
                f'<td class="val aft">{sa*100:.1f}%</td>'
                f'<td style="color:{BRAND["green"] if diff>=0 else BRAND["red"]};'
                f'font-size:.67rem;font-weight:600;">{diff_str}</td>'
                f'</tr>'
            )
        return ui.HTML(f"""
<div style="overflow-x:auto;margin-top:4px;">
  <table class="prob-tbl">
    <thead><tr>
      <th>Time</th>
      <th style="color:{_COX_CLR};">Cox PH</th>
      <th style="color:{_AFT_CLR};">AFT</th>
      <th>AFT - Cox</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>
<p style="font-size:.64rem;color:{_MUTED};margin-top:8px;line-height:1.42;">
  The rightmost column shows the AFT estimate minus the Cox PH estimate at each time point.<br>
  These model-based estimates were derived from TCGA-LUAD data and have not been clinically validated for individual prognosis.
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
        # Function guide: _fmt is responsible for perform the routine-specific application step.
        # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
        # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
        def _fmt(v, decimals=3):
            return f"{v:.{decimals}f}" if isinstance(v, float) and not np.isnan(v) else "—"

        # Function guide: _chip is responsible for perform the routine-specific application step.
        # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
        # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
        def _chip(label, train_v, test_v, clr, decimals=3):
            _tint = "#F0F2F5" if clr == _COX_CLR else "#EEF6F4"
            return (
                f'<div class="metric-chip" style="--tile:{clr};--tint:{_tint};">'
                f'<span class="mc-label">{label}</span>'
                f'<span class="mc-value" style="color:{clr};">'
                f'{_fmt(train_v, decimals)} / {_fmt(test_v, decimals)}</span>'
                f'<span style="font-size:.60rem;color:{_MUTED};">training / test</span>'
                f'</div>'
            )

        chips = "".join([
            _chip("Cox PH — C-index",
                  TR_COX.get("c_index", float("nan")), RES_COX["c_index"], _COX_CLR),
            _chip("Cox PH — IBS",
                  TR_COX.get("ibs", float("nan")), RES_COX["ibs"], _COX_CLR, 4),
            f'<div class="metric-chip" style="--tile:{_COX_CLR};--tint:#F0F2F5;">'
            f'<span class="mc-label">Cox PH — mean AUC</span>'
            f'<span class="mc-value mc-cox">{_fmt(RES_COX["mean_auc"])}</span>'
            f'<span style="font-size:.60rem;color:{_MUTED};">test only</span>'
            f'</div>',
            _chip("AFT — C-index",
                  TR_AFT.get("c_index", float("nan")), RES_AFT["c_index"], _AFT_CLR),
            _chip("AFT — IBS",
                  TR_AFT.get("ibs", float("nan")), RES_AFT["ibs"], _AFT_CLR, 4),
            f'<div class="metric-chip" style="--tile:{_AFT_CLR};--tint:#EEF6F4;">'
            f'<span class="mc-label">AFT — mean AUC</span>'
            f'<span class="mc-value mc-aft">{_fmt(RES_AFT["mean_auc"])}</span>'
            f'<span style="font-size:.60rem;color:{_MUTED};">test only</span>'
            f'</div>',
        ])
        return ui.HTML(
            f'<div class="infobar" style="margin-top:10px;">{chips}</div>'
        )

    # ── Methods panel ─────────────────────────────────────────────────────────

    @render.ui
    # Function guide: methods_panel is responsible for perform the routine-specific application step.
    # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def methods_panel():
        feat_rows = "".join(
            f"<tr><td><code>{c}</code></td>"
            f"<td>{FEAT_DISPLAY[c]}</td>"
            f"<td class='num'>{float(_IMP.statistics_[i]):.1f}</td></tr>"
            for i, c in enumerate(FEAT_COLS)
        )

        # Function guide: _f is responsible for perform the routine-specific application step.
        # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
        # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
        def _f(v, d=3):
            return f"{v:.{d}f}" if isinstance(v, float) and not np.isnan(v) else "—"

        cmp_rows = ""
        best_c = max(RES_COX["c_index"], RES_AFT["c_index"])
        for nm, res, tr, clr in [
            ("Cox PH", RES_COX, TR_COX, _COX_CLR),
            (f"{DIST['best']} AFT", RES_AFT, TR_AFT, _AFT_CLR),
        ]:
            is_best = abs(res["c_index"] - best_c) < 1e-9
            bold = f"font-weight:700;color:{clr};" if is_best else ""
            star = " ★" if is_best else ""
            cmp_rows += (
                f'<tr style="{bold}">'
                f"<td>{nm}{star}</td>"
                f"<td class='num'>{_f(tr.get('c_index', float('nan')))} / {_f(res['c_index'])}</td>"
                f"<td class='num'>[{_f(res['ci_lo'])}, {_f(res['ci_hi'])}]</td>"
                f"<td class='num'>{_f(tr.get('ibs', float('nan')), 4)} / {_f(res['ibs'], 4)}</td>"
                f"<td class='num'>{_f(res['mean_auc'])}</td>"
                f"</tr>"
            )

        return ui.HTML(f"""
<div class="methods">

  <div class="card" style="margin-bottom:14px;">
    <div class="card-header">Model Comparison — Training and Test Sets
      &nbsp;<span style="font-weight:400;font-size:.78em;color:{_MUTED};">
        Training N = {N_TRAIN} · Test N = {N_TEST} · ★ = highest test C-index</span>
    </div>
    <div class="card-body" style="padding:14px!important;">
      <table class="mtbl">
        <thead><tr>
          <th>Model</th><th>C-index (training / test)</th><th>95% Bootstrap CI</th>
          <th>IBS (training / test)</th><th>Mean AUC (test)</th>
        </tr></thead>
        <tbody>{cmp_rows}</tbody>
      </table>
      <p style="font-size:.74rem;color:{_MUTED};margin-top:8px;">
        Training metrics are apparent performance (no resampling).
        A large difference between training and test performance may indicate overfitting.
      </p>
    </div>
  </div>

  <div class="card" style="margin-bottom:14px;">
    <div class="card-header">Methods</div>
    <div class="card-body method-narrative" style="padding:14px!important;">
      <section class="method-section">
      <h4>Data</h4>
      <p>The TCGA-LUAD cohort represents lung adenocarcinoma cases from The Cancer
      Genome Atlas. Data for {N_TOTAL} patients were obtained from the public NCI
      Genomic Data Commons API. Overall survival was defined as time from diagnosis
      to death or last follow-up and converted from days to months. The observed
      event rate was {EV_RATE:.0%}, and median follow-up was {MED_FU:.0f} months.
      Data were divided using an 80/20 split stratified by event status
      (training N = {N_TRAIN}; test N = {N_TEST}). Missing values were imputed using
      the training median for age and training modes for stage categories.</p>
      </section>

      <section class="method-section">
      <h4>Distribution selection</h4>
      <p>Four parametric families (Weibull, Log-Normal, Log-Logistic, Exponential) were
      fitted to the marginal survival times using maximum likelihood with censoring.
      AIC was used to select the AFT family. Best fit: <strong>{DIST['best']}</strong>
      (AIC advantage over Weibull:
      {DIST['table'].set_index('Distribution').loc['Weibull','AIC'] -
       DIST['table'].set_index('Distribution').loc[DIST['best'],'AIC']:.1f} points).</p>
      </section>

      <section class="method-section">
      <h4>Cox Proportional Hazards</h4>
      <p>A semiparametric Cox model was fitted with lifelines. The L2 penalizer was tuned by
      5-fold event-stratified CV grid search over [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0].
      Imputation, functional-form screening, spline knots and category encodings
      were fitted separately within each training fold.
      The selected penalizer was {COX._penalizer_used}. Survival curves were derived
      using the Breslow baseline-hazard estimator.</p>
      </section>

      <section class="method-section">
      <h4>{DIST['best']} Accelerated Failure Time</h4>
      <p>A fully parametric AFT model was fitted with lifelines. The AFT framework
      models survival time directly: log(T) = Xβ + σε, where ε follows the
      {DIST['best']} error distribution. Coefficients act on survival time rather than
      the hazard. The closed-form survival function permits model-based extrapolation,
      but estimates beyond the observed follow-up period require caution.</p>
      </section>

      <section class="method-section">
      <h4>Evaluation</h4>
      <p>Training-set C-index and IBS are reported as apparent performance. Test-set
      C-index is accompanied by a 95% bootstrap confidence interval based on 150
      resamples. IBS is evaluated at 12, 24, 36, 48, and 60 months. Test-set
      time-dependent AUC uses the cumulative/dynamic definition. Evaluation horizons
      lie within both follow-up ranges, with the same IBS horizons for train and test.
      Outcomes beyond the last horizon are administratively censored for test IPCW metrics.</p>
      </section>

      <section class="method-section">
      <h4>Model assumptions</h4>
      <p>Training-only Cox likelihood-ratio tests selected a 3-knot age spline and
      dummy encoding for overall and N stage; T stage retains integer coding.
      Proportional-hazards diagnostics were reviewed on training data only.
      Overall stage overlaps with T, N, and M information. Missing M stage
      is common, and simple imputation does not represent its full uncertainty.
      External validation is needed before clinical use.</p>
      </section>
    </div>
  </div>

  <div class="card">
    <div class="card-header">Features ({len(FEAT_COLS)} clinical variables)</div>
    <div class="card-body" style="padding:14px!important;">
      <table class="mtbl">
        <thead><tr>
          <th>Column</th><th>Description</th><th>Imputation Value</th>
        </tr></thead>
        <tbody>{feat_rows}</tbody>
      </table>
      <p style="font-size:.76rem;color:{_MUTED};margin-top:10px;">
        Missing age uses the training median; missing stage categories use training modes.
        Stage distribution: I={int(STAGE_COUNTS.get("1", 0))}
        · II={int(STAGE_COUNTS.get("2", 0))}
        · III={int(STAGE_COUNTS.get("3", 0))}
        · IV={int(STAGE_COUNTS.get("4", 0))}.
      </p>
    </div>
  </div>

</div>
""")


app = App(app_ui, server)
