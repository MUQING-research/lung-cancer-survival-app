"""Focused regression checks for the survival deployment contract."""

import copy
import pickle
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sksurv.metrics import concordance_index_censored

import survival_core as sc
import survival_app as app


class SurvivalContractTests(unittest.TestCase):
    def test_source_cleaning_excludes_unknown_outcome_and_nonpositive_time(self):
        diagnosis = {"days_to_last_follow_up": 100, "age_at_diagnosis": 20000, "ajcc_pathologic_stage": "Stage IA"}
        known = {"case_id": "known", "demographic": {"vital_status": "Alive"}, "diagnoses": [diagnosis]}
        unknown = {"case_id": "unknown", "demographic": {}, "diagnoses": [diagnosis]}
        zero = {"case_id": "zero", "demographic": {"vital_status": "Dead", "days_to_death": 0}, "diagnoses": [diagnosis]}
        cleaned = app._preprocess([known, known, unknown, zero])
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned.attrs["cleaning_audit"]["excluded"], {"duplicate_case": 1, "unknown_vital_status": 1, "invalid_survival_time": 1})

    def test_training_split_and_statistics_match_bundle(self):
        if not app._GDC_CACHE.exists():
            self.skipTest("Local raw cache intentionally absent in deployment.")
        source = app._preprocess(app._download_gdc())
        train, test = train_test_split(source, test_size=0.2, random_state=sc.SEED, stratify=source[sc.EVENT_COL])
        self.assertTrue(set(train.case_id).isdisjoint(test.case_id))
        self.assertEqual((len(train), len(test)), (app.N_TRAIN, app.N_TEST))
        expected = sc.ClinicalImputer().fit(train[app.FEAT_COLS]).statistics_
        np.testing.assert_allclose(app._IMP.statistics_, expected)
        before = pickle.dumps(app._IMP)
        changed_test = test[app.FEAT_COLS].copy()
        changed_test["age"] = 99
        app._IMP.transform(changed_test)
        self.assertEqual(before, pickle.dumps(app._IMP))

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

    def test_ipcw_late_events_do_not_invalidate_supported_horizons(self):
        train = pd.DataFrame({sc.TIME_COL: np.arange(1, 11, dtype=float), sc.EVENT_COL: [1, 0] * 5})
        test = pd.DataFrame({sc.TIME_COL: [1.5, 3.5, 4.5, 50., 80.], sc.EVENT_COL: [1, 1, 0, 1, 0]})
        raw = pd.DataFrame({"age": [45, 55, 65, 75, 80], "stage": [1, 2, 3, 4, 2], "t_stage": [1, 2, 3, 4, 2], "n_stage": [0, 1, 2, 3, 1], "m_stage": [0, 0, 0, 1, 0]})
        features = app._IMP.transform(raw)
        result = sc.evaluate(app.COX, "cox", test, app._make_y(train), app._make_y(test),
                             times=np.array([2., 4., 6.]), feat_df_test=features, n_boot=150)
        self.assertEqual(result["metric_errors"], {})
        self.assertTrue(np.isfinite(result["ibs"]))
        expected = concordance_index_censored(test[sc.EVENT_COL].astype(bool), test[sc.TIME_COL], result["risk"])[0]
        self.assertAlmostEqual(result["c_index"], expected)
        with self.assertRaisesRegex(ValueError, "two supported"):
            sc.evaluate(app.COX, "cox", test, app._make_y(train), app._make_y(test), times=np.array([60.]), feat_df_test=features)

    def test_bundle_allowlist_and_roundtrip_predictions(self):
        raw = pd.DataFrame([{ "age": 65, "stage": 2, "t_stage": 2, "n_stage": 0, "m_stage": 0}])
        features = app._IMP.transform(raw)
        before = app.COX.predict_survival_function(features, times=[12, 24, 60]).to_numpy()
        polluted = copy.deepcopy(app._B)
        polluted["df_train"] = raw
        polluted["feat_test"] = features
        polluted["cox"]._model._training_data = features
        clean, changed = app._sanitized_bundle(polluted)
        self.assertTrue(changed)
        self.assertNotIn("df_train", clean)
        self.assertNotIn("feat_test", clean)
        self.assertNotIn("_training_data", vars(clean["cox"]._model))
        restored = pickle.loads(pickle.dumps(clean))
        np.testing.assert_allclose(restored["cox"].predict_survival_function(features, times=[12, 24, 60]), before)
        for model in ("cox", "aft"):
            self.assertTrue(np.isfinite(clean[f"tr_{model}"]["ibs"]))
            self.assertTrue(np.isfinite(clean[f"res_{model}"]["ibs"]))
            self.assertGreaterEqual(clean[f"res_{model}"]["bootstrap_successful"], 150)
            self.assertEqual(clean[f"tr_{model}"]["times"], clean[f"res_{model}"]["times"])

    def test_clinical_inputs_reject_invalid_values(self):
        valid = {"age": 65, "stage": 2, "t_stage": 2, "n_stage": 0, "m_stage": 0}
        self.assertEqual(sc.validate_clinical_inputs(valid)["age"], 65)
        for field, bad in [("age", None), ("age", ""), ("age", -1), ("age", 101),
                           ("age", float("nan")), ("age", float("inf")), ("stage", 5), ("n_stage", 0.5)]:
            with self.subTest(field=field, value=bad), self.assertRaises(ValueError):
                sc.validate_clinical_inputs({**valid, field: bad})


if __name__ == "__main__":
    unittest.main(verbosity=2)
