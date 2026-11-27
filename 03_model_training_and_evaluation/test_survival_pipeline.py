# Module guide:
# - Role: Train, evaluate, or test the survival modelling workflow.
# - Workflow: Compare Cox, AFT, and related survival utilities on train and test partitions and validate visualizations.
# - Design note: Keep regression tests focused on reproducible bundle contents and evaluation contracts.
"""Focused regression checks for the survival training and deployment pipeline."""

import copy
import pickle
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sksurv.metrics import concordance_index_censored

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_model_deployment"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_data_preprocessing"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_data_cleaning"))
import survival_core as sc
import survival_app as app


# Class guide: SurvivalPipelineTests is responsible for perform the requested operation.
# Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
class SurvivalPipelineTests(unittest.TestCase):
    # Function guide: test_source_cleaning_excludes_unknown_outcome_and_nonpositive_time is responsible for test source cleaning excludes unknown outcome and nonpositive time.
    # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def test_source_cleaning_excludes_unknown_outcome_and_nonpositive_time(self):
        diagnosis = {"days_to_last_follow_up": 100, "age_at_diagnosis": 20000, "ajcc_pathologic_stage": "Stage IA"}
        known = {"case_id": "known", "demographic": {"vital_status": "Alive"}, "diagnoses": [diagnosis]}
        unknown = {"case_id": "unknown", "demographic": {}, "diagnoses": [diagnosis]}
        zero = {"case_id": "zero", "demographic": {"vital_status": "Dead", "days_to_death": 0}, "diagnoses": [diagnosis]}
        cleaned = app._clean_clinical_records([known, known, unknown, zero])
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(
            cleaned.attrs["cleaning_audit"]["excluded_rows"],
            {
                "duplicate_case": 1,
                "unknown_vital_status": 1,
                "invalid_survival_time": 1,
            },
        )

    # Function guide: test_training_split_and_statistics_match_bundle is responsible for test training split and statistics match bundle.
    # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def test_training_split_and_statistics_match_bundle(self):
        if not app._GDC_CACHE.exists():
            self.skipTest("Local raw cache intentionally absent in deployment.")
        source = app._clean_clinical_records(app._download_gdc())
        train, test = train_test_split(source, test_size=0.2, random_state=sc.SEED, stratify=source[sc.EVENT_COL])
        self.assertTrue(set(train.case_id).isdisjoint(test.case_id))
        self.assertEqual((len(train), len(test)), (app.N_TRAIN, app.N_TEST))
        self.assertEqual(app._IMP.n_imputations, app.N_IMPUTATIONS)
        self.assertEqual(len(app._IMP.preprocessors_), app.N_IMPUTATIONS)
        before = pickle.dumps(app._IMP)
        changed_test = test[app.FEAT_COLS].copy()
        changed_test["age"] = 99
        transformed = app._IMP.transform_multiple(changed_test)
        self.assertEqual(len(transformed), app.N_IMPUTATIONS)
        self.assertTrue(all(np.isfinite(frame.to_numpy()).all() for frame in transformed))
        self.assertEqual(before, pickle.dumps(app._IMP))

    # Function guide: test_test_metrics_use_training_fitted_imputation is responsible for test metrics use training fitted imputation.
    # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def test_test_metrics_use_training_fitted_imputation(self):
        self.assertEqual(app.N_TEST_EVAL, app.N_TEST)
        self.assertEqual(
            app._B["provenance"]["test_non_evaluable_missing"],
            0,
        )
        self.assertGreater(app._B["provenance"]["test_rows_with_missing_predictors"], 0)
        self.assertEqual(app._B["res_cox"]["surv_mat"].shape[0], app.N_TEST)
        self.assertIn("apply the fitted transformer", app._B["provenance"]["cleaning"]["missing_predictor_rule"])

    # Function guide: test_all_categorical_features_are_dummy_encoded is responsible for verify categorical design columns.
    # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def test_all_categorical_features_are_dummy_encoded(self):
        for preprocessor in app._IMP.preprocessors_:
            self.assertEqual(
                preprocessor.forms_["stage"], "dummy"
            )
            self.assertEqual(
                preprocessor.forms_["t_stage"], "dummy"
            )
            self.assertEqual(
                preprocessor.forms_["n_stage"], "dummy"
            )
            self.assertEqual(
                preprocessor.forms_["m_stage"], "dummy"
            )
            dummy_columns = {column for column in preprocessor.output_columns_
                             if column.startswith(("stage_", "t_stage_", "n_stage_", "m_stage_"))}
            expected = {
                f"{name}_{int(level)}"
                for name, levels in preprocessor.categories_.items()
                for level in levels[1:]
            }
            self.assertEqual(dummy_columns, expected)
        age_forms = {preprocessor.forms_["age"] for preprocessor in app._IMP.preprocessors_}
        self.assertEqual(age_forms, {app._IMP.forms_["age"]})
        self.assertIn(app._IMP.forms_["age"], {"linear", "rcs"})
        age_summary = next(
            item for item in app._B["diagnostics"]["functional_form_checks"]
            if item.get("method") == "Multiple-imputation article-style Cox RCS Wald tests"
        )
        self.assertEqual(age_summary["n_imputations"], app.N_IMPUTATIONS)
        self.assertIn(age_summary["decision"], {"linear", "rcs"})
        self.assertEqual(len(age_summary["p_total_values"]), app.N_IMPUTATIONS)
        self.assertEqual(len(age_summary["p_nonlinear_values"]), app.N_IMPUTATIONS)
        for check in app._IMP.form_checks_[1:]:
            self.assertEqual(
                check["method"],
                "Article-style Cox RCS overall and nonlinear Wald tests",
            )
            self.assertIn("p_total", check)
            self.assertIn("p_nonlinear", check)
        self.assertEqual(
            {item["variable"] for item in app._B["diagnostics"]["functional_form_checks"]},
            {"age"},
        )
        self.assertEqual(
            {item["variable"] for item in app._B["diagnostics"]["categorical_encoding_checks"]},
            {"stage", "t_stage", "n_stage", "m_stage"},
        )
        self.assertEqual(
            app._B["diagnostics"]["feature_form_figures"],
            {"age": "feature_form_age.png"},
        )

    # Function guide: test_cv_fits_preprocessing_on_folds_only is responsible for test cv fits preprocessing on folds only.
    # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def test_cv_fits_preprocessing_on_folds_only(self):
        rng = np.random.default_rng(sc.SEED)
        n = 80
        raw = pd.DataFrame({"age": rng.uniform(40, 80, n), "stage": rng.integers(1, 5, n),
                            "t_stage": rng.integers(1, 5, n), "n_stage": rng.integers(0, 4, n),
                            "m_stage": rng.integers(0, 2, n)})
        raw.loc[::4, "age"] = np.nan
        outcome = pd.DataFrame({sc.TIME_COL: rng.exponential(20, n) + 0.1, sc.EVENT_COL: np.arange(n) % 2})
        rows_seen = []
        original = sc.ClinicalPreprocessor.fit

        # Function guide: observed_fit is responsible for perform fit.
        # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
        # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
        def observed_fit(transformer, X, y):
            rows_seen.append(set(X.index))
            return original(transformer, X, y)

        initial = sc.ClinicalPreprocessor().fit(raw, outcome).transform(raw)
        with patch.object(sc.ClinicalPreprocessor, "fit", observed_fit):
            selected = sc.tune_cox_penalizer(outcome, feat_df=initial, raw_feat_df=raw)
        self.assertIn(selected, [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0])
        self.assertEqual(len(rows_seen), 5)
        self.assertTrue(all(len(indices) == 64 for indices in rows_seen))
        self.assertEqual(set.union(*rows_seen), set(raw.index))

    # Function guide: test_exponential_family_has_fixed_shape is responsible for test exponential family has fixed shape.
    # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def test_exponential_family_has_fixed_shape(self):
        rng = np.random.default_rng(sc.SEED)
        frame = pd.DataFrame({"x": rng.normal(size=160)})
        frame[sc.TIME_COL] = rng.exponential(np.exp(2 + 0.4 * frame.x)) + 0.01
        frame[sc.EVENT_COL] = 1
        fitted = sc.build_aft(frame, aft_class=sc.ExponentialAFTFitter, feat_df=frame[["x"]])
        self.assertEqual(set(fitted.params_.index.get_level_values(0)), {"lambda_"})
        survival = fitted.predict_survival_function(frame[["x"]].iloc[:3], times=[1, 2]).to_numpy()
        np.testing.assert_allclose(survival[1], survival[0] ** 2, rtol=1e-9)
        np.testing.assert_allclose(fitted.predict_median(frame[["x"]]), fitted.predict_expectation(frame[["x"]]) * np.log(2))

    # Function guide: test_ipcw_late_events_do_not_invalidate_supported_horizons is responsible for test ipcw late events do not invalidate supported horizons.
    # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def test_ipcw_late_events_do_not_invalidate_supported_horizons(self):
        train = pd.DataFrame({sc.TIME_COL: np.arange(1, 11, dtype=float), sc.EVENT_COL: [1, 0] * 5})
        test = pd.DataFrame({sc.TIME_COL: [1.5, 3.5, 4.5, 50., 80.], sc.EVENT_COL: [1, 1, 0, 1, 0]})
        raw = pd.DataFrame({"age": [45, 55, 65, 75, 80], "stage": [1, 2, 3, 4, 2], "t_stage": [1, 2, 3, 4, 2], "n_stage": [0, 1, 2, 3, 1], "m_stage": [0, 0, 0, 1, 0]})
        features = app._IMP.transform(raw)
        result = sc.evaluate(app.COX, "cox", test, app._build_survival_target(train), app._build_survival_target(test),
                             times=np.array([2., 4., 6.]), feat_df_test=features, n_boot=150)
        self.assertEqual(result["metric_errors"], {})
        self.assertTrue(np.isfinite(result["ibs"]))
        expected = concordance_index_censored(test[sc.EVENT_COL].astype(bool), test[sc.TIME_COL], result["risk"])[0]
        self.assertAlmostEqual(result["c_index"], expected)
        with self.assertRaisesRegex(ValueError, "two supported"):
            sc.evaluate(app.COX, "cox", test, app._build_survival_target(train), app._build_survival_target(test), times=np.array([60.]), feat_df_test=features)

    # Function guide: test_bundle_allowlist_and_roundtrip_predictions is responsible for test bundle allowlist and roundtrip predictions.
    # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def test_bundle_allowlist_and_roundtrip_predictions(self):
        raw = pd.DataFrame([{ "age": 65, "stage": 2, "t_stage": 2, "n_stage": 0, "m_stage": 0}])
        features = app._IMP.transform(raw)
        before = app.COX.predict_survival_function(features, times=[12, 24, 60]).to_numpy()
        polluted = copy.deepcopy(app._B)
        polluted["df_train"] = raw
        polluted["feat_test"] = features
        polluted["cox"].models[0]._training_data = features
        clean, changed = app._sanitized_bundle(polluted)
        self.assertTrue(changed)
        self.assertNotIn("df_train", clean)
        self.assertNotIn("feat_test", clean)
        self.assertNotIn("_training_data", vars(clean["cox"].models[0]))
        restored = pickle.loads(pickle.dumps(clean))
        np.testing.assert_allclose(restored["cox"].predict_survival_function(features, times=[12, 24, 60]), before)
        for model in ("cox", "aft"):
            self.assertTrue(np.isfinite(clean[f"tr_{model}"]["ibs"]))
            self.assertTrue(np.isfinite(clean[f"res_{model}"]["ibs"]))
            self.assertGreaterEqual(clean[f"res_{model}"]["bootstrap_successful"], 150)
            self.assertEqual(clean[f"tr_{model}"]["times"], clean[f"res_{model}"]["times"])

    # Function guide: test_clinical_inputs_reject_invalid_values is responsible for test clinical inputs reject invalid values.
    # Inputs: the component state and explicit routine arguments. Outputs and side effects follow the routine contract.
    # Keep this boundary focused on one workflow step so it remains easy to test and reuse.
    def test_clinical_inputs_reject_invalid_values(self):
        valid = {"age": 65, "stage": 2, "t_stage": 2, "n_stage": 0, "m_stage": 0}
        self.assertEqual(sc.validate_clinical_inputs(valid)["age"], 65)
        for field, bad in [("age", -1), ("age", 101), ("age", float("inf")),
                           ("stage", 5), ("n_stage", 0.5)]:
            with self.subTest(field=field, value=bad), self.assertRaises(ValueError):
                sc.validate_clinical_inputs({**valid, field: bad})

    # Function guide: test_t0_and_missing_m_stage_follow_the_input_contract
    # verifies that T0 is accepted and missing inputs are transformed by MICE.
    def test_t0_and_missing_m_stage_follow_the_input_contract(self):
        values = sc.validate_clinical_inputs({
            "age": 65, "stage": 2, "t_stage": 0, "n_stage": 0,
            "m_stage": 0,
        })
        self.assertEqual(values["t_stage"], 0)
        transformed = app._IMP.transform_multiple(pd.DataFrame([values]))
        self.assertEqual(len(transformed), app.N_IMPUTATIONS)
        self.assertTrue(all(np.isfinite(frame.to_numpy()).all() for frame in transformed))
        for missing_value in (None, "", "unknown", float("nan")):
            with self.subTest(value=missing_value):
                missing_values = sc.validate_clinical_inputs({**values, "m_stage": missing_value})
                missing_transformed = app._IMP.transform_multiple(pd.DataFrame([missing_values]))
                self.assertTrue(all(np.isfinite(frame.to_numpy()).all() for frame in missing_transformed))


if __name__ == "__main__":
    unittest.main(verbosity=2)
