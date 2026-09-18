# Model training and evaluation

This folder contains the offline survival-model training command, deployment-bundle export, and model/preprocessing/visualization checks.

Install the same runtime dependencies and run the checks from the repository root:

```bash
python -m pip install -r 03_model_training_and_evaluation/requirements.txt
python 03_model_training_and_evaluation/train_and_export_model_bundle.py
python -m unittest discover -s 03_model_training_and_evaluation -p "test_*.py" -v
```

Five-fold cross-validation is used for Cox penalizer selection as internal validation. The fixed 20% holdout is reserved for test-set evaluation and is not used for tuning. Training and audit outputs remain outside `04_model_deployment/`. The checked-in bundle in `04_model_deployment/` is the model artifact sent to shinyapps.io.
