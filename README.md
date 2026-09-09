# TCGA-LUAD Survival Prediction Dashboard

An interactive [Shiny for Python](https://shiny.posit.co/py/) dashboard that compares Cox proportional hazards and parametric accelerated failure time models for overall survival prediction.

[Open the live application](https://medictio.shinyapps.io/nsclc-survival/)

> [!WARNING]
> This project is intended for research, education, and software demonstration only. It is not a medical device, has not been externally validated, and must not be used to guide treatment or estimate prognosis for an individual patient.

## Application preview

![Lung cancer survival dashboard](assets/lung_cancer_app.png)

## Overview

The application uses publicly available clinical data from the [TCGA Lung Adenocarcinoma project](https://portal.gdc.cancer.gov/projects/TCGA-LUAD), obtained through the NCI Genomic Data Commons API. The checked-in model bundle contains 509 evaluable patients, with an overall survival event rate of 36% and median observed follow-up of 21.6 months.

The dashboard provides:

- patient-specific Cox PH and Log-Logistic AFT survival curves;
- predicted survival probabilities at clinically interpretable time points;
- train/test C-index and Integrated Brier Score comparisons;
- test-set bootstrap confidence intervals for C-index;
- test-set cumulative/dynamic AUC over time;
- marginal survival-distribution comparison by AIC;
- cohort stage summaries and a global visitor map; and
- a deployment-safe precomputed bundle without raw clinical feature tables.

## Model workflow

```text
TCGA-LUAD clinical survival data (n = 509)
                         |
        Base cleaning and outcome construction
          overall survival measured in months
                         |
             80/20 event-stratified split
                random seed = 42
                         |
      Age median and stage-mode imputation
       Age spline, stage/N dummy encoding
                         |
          +--------------+--------------+
          |                             |
   L2-regularized Cox PH          Parametric AFT
  5-fold CV penalizer grid       family selected by AIC
  selected penalizer = 0.01      Log-Logistic selected
          |                             |
          +--------------+--------------+
                         |
       Train apparent performance + held-out test
             C-index, IBS, and dynamic AUC
                         |
       Saved bundle -> Shiny survival dashboard
```

The fixed split contains 407 training patients and 102 held-out test patients. Imputation, functional-form screening, spline construction, Cox penalizer tuning, parametric-family selection, and model fitting use the training split only. Preprocessing and functional-form decisions are refitted inside every Cox cross-validation training fold.

### Predictors

The current `tcga_luad_app_bundle.pkl` uses five clinical predictors:

1. age at diagnosis;
2. AJCC pathologic stage;
3. pathologic T category;
4. pathologic N category; and
5. pathologic M category.

Missing age is imputed with the training median; missing stage categories use training modes. Training-only Cox partial-likelihood LRTs select a 3-knot cubic spline for age and reference-level dummy coding for overall stage and N category. T category remains integer-coded and M remains binary. The fitted transformations and reference categories are saved in `eda_decisions.json`. Users must preserve the coding and units expected by the application.

## Outcome and prediction semantics

The outcome is overall survival from diagnosis to death or last follow-up, converted from days to months. A recorded death is the event; patients alive at last follow-up are right-censored.

The application compares two model families:

- **Cox PH:** a semiparametric proportional-hazards model with L2 regularization and a Breslow baseline-hazard estimator.
- **Log-Logistic AFT:** a parametric accelerated failure time model selected by the lowest training-set AIC among Weibull, Log-Normal, Log-Logistic, and Exponential candidates.

Displayed values such as `S(24 months)` are predicted probabilities of remaining alive beyond the specified time. They are population-model estimates rather than guarantees for an individual patient.

## Current bundle performance

The following values describe the checked-in bundle and its fixed 80/20 split. Training metrics are apparent performance; test metrics are held-out estimates. Mean AUC is reported for the test set only.

| Model | C-index train | C-index test | Test 95% bootstrap CI | IBS train | IBS test | Mean test AUC |
|---|---:|---:|---:|---:|---:|---:|
| Cox PH | 0.679 | 0.691 | 0.602-0.768 | 0.1863 | 0.1903 | 0.728 |
| Log-Logistic AFT | 0.679 | 0.689 | 0.602-0.764 | 0.1871 | 0.1898 | 0.729 |

Test C-index confidence intervals use 150 bootstrap resamples. Integrated Brier Score and cumulative/dynamic AUC are evaluated at 12, 24, 36, 48, and 60 months after restricting evaluation times to the observed support of both splits.

For test IPCW metrics, follow-up beyond the last supported evaluation horizon is administratively censored just after that horizon. This preserves earlier case/control status and prevents irrelevant late events from exceeding training censoring support. The original outcomes are retained for C-index and bootstrap intervals. Runtime and model checks can be reproduced with `python -m unittest validate_model -v`; the source-cache check is skipped when raw data is intentionally absent.

### Model evaluation figures

All figures below are rendered from `tcga_luad_app_bundle.pkl`. No raw clinical feature records are required to reproduce them. They use the shared Cell red/blue/teal palette with navy typography and can be regenerated with `python generate_readme_figures.py`.

#### Discrimination and prediction error

Panel A compares apparent training C-index with held-out test C-index and its bootstrap interval. Panel B compares train/test integrated Brier scores. Panel C shows held-out cumulative/dynamic AUC. The time-specific Brier score curve is intentionally omitted.

![Train and test concordance, integrated Brier score, and time-dependent AUC](assets/model_performance.png)

#### Parametric distribution selection

The training-set Kaplan-Meier curve is compared with four marginal parametric survival distributions. Log-Logistic had the lowest AIC (1579.1) and was therefore used for the covariate-adjusted AFT model. Exact AIC and BIC values remain available in the application table.

![Kaplan-Meier curve and fitted parametric survival distributions](assets/marginal_survival_fit.png)

## Run locally

Use Python 3.13 (the bundles and deployment use Python 3.13.9).

```bash
git clone https://github.com/MUQING-research/lung-cancer-survival-app.git
cd lung-cancer-survival-app

python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS or Linux
source .venv/bin/activate
```

Install dependencies and start the app:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
shiny run --reload app.py
```

Open the local URL printed by Shiny, normally `http://127.0.0.1:8000`.

The app loads `tcga_luad_app_bundle.pkl` at startup. Keep the bundle beside `survival_app.py` when packaging or deploying the application.

## Deployment bundle

The tracked bundle contains fitted models, imputation statistics, outcome labels, survival probability matrices, evaluation metrics, marginal distribution fits, and aggregate cohort summaries. It does not contain raw training or test feature tables.

If the bundle is absent, the application can retrieve TCGA-LUAD clinical data from the GDC API, fit the models, and create the bundle. Model training is intentionally kept outside normal deployed startup by checking in the sanitized bundle.

## Deployment

The repository includes a shinyapps.io deployment helper:

```bash
python -m pip install rsconnect-python
python deploy.py
```

Configure `rsconnect` credentials before running the helper from the Python 3.13 environment matching `requirements.txt`. Run `python deploy.py --check` for a local preflight. The helper updates the existing `medictio/nsclc-survival` app, checks pinned package versions, and uploads only the runtime file allowlist, including the precomputed model bundle and stylesheet.

The pinned `scikit-survival==0.25.0` supports `scikit-learn==1.7.2`; this compatibility is documented in the [versioned installation guide](https://scikit-survival.readthedocs.io/en/v0.25.0/install.html).

For another hosting workflow, package at least:

- `app.py`
- `survival_app.py`
- `survival_core.py`
- `tcga_luad_app_bundle.pkl`
- `eda_decisions.json`
- `requirements.txt`
- `theme.css`
- `world.geojson` if the visitor map is enabled

Do not replace the deployment bundle with raw patient-level feature tables.

## Optional visitor analytics

The dashboard can optionally render aggregate visit statistics through Supabase and locate public IP addresses through IPinfo or the `ipwho.is` fallback. Remote persistence requires both `SUPABASE_URL` and `SUPABASE_KEY`; without them, the app uses a bounded process-local visit list that resets on restart. Hosted visitors should be informed when location lookup is enabled. Prediction inputs are not stored.

The shinyapps.io deployment helper does not forward environment variables: that platform does not support `rsconnect --environment` management. Container hosts can inject the variables below. See the [Posit deployment documentation](https://docs.posit.co/rsconnect-python/deploying/).

| Environment variable | Purpose |
|---|---|
| `IPINFO_TOKEN` | Optional authenticated IPinfo lookup |
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_KEY` | Supabase client key; restrict access with row-level security |

## Repository layout

```text
.
|-- app.py                         # Minimal Shiny entry point
|-- survival_app.py                # Data handling, UI, server, and figures
|-- survival_core.py               # Survival models and evaluation helpers
|-- generate_readme_figures.py     # Reproducible README figure generator
|-- tcga_luad_app_bundle.pkl       # Sanitized model and evaluation bundle
|-- eda_decisions.json             # Training-only preprocessing decisions and provenance
|-- bundle_survival.py             # Canonical offline rebuild entry point
|-- validate_model.py              # Model, bundle, and IPCW regression checks
|-- theme.css                      # Active application stylesheet
|-- assets/                        # App preview and model figures for this README
|-- requirements.txt               # Runtime dependencies
|-- deploy.py                      # shinyapps.io deployment helper
|-- Dockerfile                     # Container definition
|-- upload.py                      # Hugging Face Space upload helper
`-- world.geojson                  # Basemap used by visitor analytics
```

## Limitations

- This is a retrospective analysis of one public cancer cohort and has no external, temporal, or prospective validation.
- The held-out test set contains only 102 patients, producing wide C-index intervals and limited precision for late-horizon estimates.
- Only five routinely recorded clinical predictors are used; treatment, molecular, imaging, comorbidity, and performance-status information are not included.
- Proportional-hazards assumptions and parametric extrapolation assumptions require further validation before any clinical use.
- The single fixed split does not quantify the full variability of model development and selection.
- Median/mode imputation does not capture uncertainty from missing clinical information, particularly the frequently missing M category.
- Predictions reflect the TCGA-LUAD cohort and should not be generalized to other lung-cancer histologies or care settings without validation.

## Data attribution

The Cancer Genome Atlas Research Network. *Lung Adenocarcinoma (TCGA-LUAD)*. Clinical data are accessed through the [NCI Genomic Data Commons](https://portal.gdc.cancer.gov/projects/TCGA-LUAD).
