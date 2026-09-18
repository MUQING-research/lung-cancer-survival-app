# Data cleaning

This stage downloads the TCGA-LUAD clinical response from the NCI GDC API, removes duplicate case identifiers and unusable survival outcomes, standardizes stage coding, constructs overall-survival time and event fields, and writes a data dictionary plus cleaning audit.

It does not fit imputation, scaling, functional-form, or model parameters. Those operations begin after the train/test split in stage 02.

```bash
python 01_data_cleaning/clean_clinical_data.py
```

Outputs are written to `03_model_training_and_evaluation/.cache/training/` and are excluded from deployment and Git.
