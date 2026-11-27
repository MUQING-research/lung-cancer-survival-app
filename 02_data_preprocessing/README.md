# Survival preprocessing

This stage builds the structured event-time target and fits training-only clinical preprocessing. Five posterior-sampling MICE imputations are fitted on the training predictors; age remains continuous and categorical clinical codes are rounded and clipped to their documented domains. Each completed training dataset fits an adjusted three-knot RCS Cox model and reports the article-style overall P and P nonlinear tests. The majority decision (linear or RCS) is applied consistently across the imputed models. All categorical clinical codes use treatment-coded dummy variables.

The fitted imputation and functional-form preprocessors are applied unchanged to validation, held-out test, and deployment rows; no preprocessing parameter is refitted on the test set. The implementation is `survival_preprocessing.py`; the survival-specific estimator classes remain in the deployment-compatible `survival_core.py` module.
