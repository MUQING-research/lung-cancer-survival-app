# Survival preprocessing

This stage builds the structured event-time target and fits training-only clinical preprocessing. Age uses median imputation; categorical clinical variables use training-set modes. Cox partial-likelihood checks select the functional form and encoding, with a cubic spline considered for age and dummy coding considered for ordered categories.

The fitted preprocessing object and decisions are applied unchanged to the test set. The implementation is `survival_preprocessing.py`; the survival-specific estimator classes remain in the deployment-compatible `survival_core.py` module.
