import json
import unittest
from unittest.mock import patch

import pandas as pd

from scripts.service import benchmark, ml_benchmark
from src.benchmarking import group_k_fold_indices, group_k_fold_test_groups


def _response_json(*, included_state_ids: list[str] | None = None) -> str:
    return json.dumps(
        {
            "inputs": {
                "variant": "1.0.0",
                "state": {"feature": 1.0},
                "exclude_uids": [],
            },
            "outputs": {
                "g1": {
                    "location_likelihood": {"l1": 1.0},
                    "per_location": {
                        "l1": {
                            "summary": {
                                "cct_weighted": 1.5,
                                "stats": {
                                    "weight_mass": 1.0,
                                    "weight_mass_mean": 1.0,
                                    "cct_weighted_std": None,
                                    "cct_distance_correlation": None,
                                    "cct_quantiles": None,
                                    "neighborhood_compactness": 1.0,
                                    "n": 1,
                                    "n_eff": 1.0,
                                    "n_unique_states": 1,
                                    "distances": {
                                        "min": 0.1,
                                        "mean": 0.1,
                                        "median": 0.1,
                                        "spread": 0.0,
                                        "norm": 1.0,
                                    },
                                },
                            },
                            "included_state_ids": included_state_ids or [],
                            "per_neighbor": [],
                        }
                    },
                    "included_state_ids": included_state_ids or [],
                    "neighbors": [],
                }
            },
        }
    )


class _FakeResponse:
    status_code = 200

    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self) -> None:
        return None


class _FakeClient:
    def __init__(self, response_json: str, requests: list[dict]):
        self.response = _FakeResponse(response_json)
        self.requests = requests

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        return None

    def post(self, endpoint: str, *, json: dict) -> _FakeResponse:
        self.requests.append({"endpoint": endpoint, "json": json})
        return self.response


class _IdentityScaler:
    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.copy()


class GroupKFoldTests(unittest.TestCase):
    def test_fold_groups_are_disjoint_exhaustive_and_match_indices(self):
        groups = pd.Series(["a", "a", "a", "b", "b", "c", "d", "d", "e"])

        indices = group_k_fold_indices(groups, n_splits=3)
        test_groups = group_k_fold_test_groups(groups, n_splits=3)

        self.assertEqual(
            test_groups,
            [frozenset(groups.iloc[test_idx]) for _, test_idx in indices],
        )
        self.assertEqual(set().union(*test_groups), set(groups))
        for fold, held_out in enumerate(test_groups):
            other_groups = set().union(*(groups for i, groups in enumerate(test_groups) if i != fold))
            self.assertTrue(held_out.isdisjoint(other_groups))


class ServiceBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.state = pd.Series({"feature": 1.0})
        self.tsa_subset = pd.DataFrame([{"CCT": 1.5, "Location": "L1", "Crit_gen": "G1"}])

    def test_process_state_sends_complete_fold_exclusion_and_records_fold(self):
        requests: list[dict] = []
        client = _FakeClient(_response_json(), requests)

        with patch.object(benchmark.httpx, "Client", return_value=client):
            reports = benchmark._process_state(
                "s1",
                self.state,
                self.tsa_subset,
                exclude_uids={"s1", "s2", "s3"},
                fold=2,
            )

        self.assertEqual(requests[0]["endpoint"], "http://localhost:8000/api/v1/estimate/tsa/by-generator")
        self.assertEqual(requests[0]["json"]["exclude_uids"], ["s1", "s2", "s3"])
        self.assertEqual(reports[0]["fold"], 2)

    def test_process_state_rejects_any_excluded_neighbor(self):
        client = _FakeClient(_response_json(included_state_ids=["s2"]), [])

        with (
            patch.object(benchmark.httpx, "Client", return_value=client),
            self.assertRaisesRegex(RuntimeError, "Data leakage"),
        ):
            benchmark._process_state(
                "s1",
                self.state,
                self.tsa_subset,
                exclude_uids={"s1", "s2"},
                fold=0,
            )

    def test_group_k_fold_analyzer_routes_each_state_with_its_complete_fold(self):
        lf = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0]}, index=["s1", "s2", "s3", "s4"])
        tsa = pd.DataFrame(
            {
                "state": ["s1", "s2", "s3", "s4"],
                "CCT": [1.0, 2.0, 3.0, 4.0],
                "Location": ["l1", "l1", "l2", "l2"],
                "Crit_gen": ["g1", "g1", "g2", "g2"],
            }
        )
        calls: list[dict] = []

        def capture_call(state_id, state, tsa_subset, *, exclude_uids, fold):
            calls.append(
                {
                    "state_id": state_id,
                    "exclude_uids": frozenset(exclude_uids),
                    "fold": fold,
                }
            )
            return []

        expected_folds = group_k_fold_test_groups(tsa["state"], n_splits=2)
        with patch.object(benchmark, "_process_state", side_effect=capture_call):
            reports = benchmark.analyze_group_k_fold(lf, tsa, n_splits=2)

        self.assertEqual(reports, [])
        self.assertEqual(len(calls), len(lf))
        self.assertEqual({call["state_id"] for call in calls}, set(lf.index))
        for call in calls:
            expected_exclusions = expected_folds[call["fold"]]
            self.assertIn(call["state_id"], expected_exclusions)
            self.assertEqual(call["exclude_uids"], expected_exclusions)

    def test_group_k_fold_payload_uses_global_fallback_and_reports_coverage(self):
        predictions = [
            {
                "fold": 0,
                "state_norm": "s1",
                "cct_true": 1.0,
                "cct_weighted_per_location": 1.0,
                "cct_weighted_global": 9.0,
            },
            {
                "fold": 0,
                "state_norm": "s2",
                "cct_true": 2.0,
                "cct_weighted_per_location": None,
                "cct_weighted_global": 2.0,
            },
            {
                "fold": 1,
                "state_norm": "s3",
                "cct_true": 3.0,
                "cct_weighted_per_location": None,
                "cct_weighted_global": None,
            },
            {
                "fold": 1,
                "state_norm": "s4",
                "cct_true": 4.0,
                "cct_weighted_per_location": 4.0,
                "cct_weighted_global": 8.0,
            },
        ]

        payload = benchmark.build_group_k_fold_payload(predictions, n_splits=2)
        # build_group_k_fold_payload reports two model rows per fold - "strict" (only the true
        # location's own coverage) and "then_global" (additionally falls back to the top-likelihood
        # location) - see its docstring. set_index("fold") alone would leave a non-unique index
        # (two rows per fold value), so index on (fold, model) to disambiguate.
        results = payload["results"].set_index(["fold", "model"])

        self.assertEqual(results.loc[(0, "service_location_then_global"), "coverage"], 1.0)
        self.assertEqual(results.loc[(0, "service_location_then_global"), "mae"], 0.0)
        self.assertEqual(results.loc[(1, "service_location_then_global"), "coverage"], 0.5)
        self.assertEqual(results.loc[(1, "service_location_then_global"), "mae"], 0.0)
        self.assertEqual(results.loc[(0, "service_location_strict"), "coverage"], 0.5)
        self.assertEqual(results.loc[(1, "service_location_strict"), "coverage"], 0.5)
        self.assertEqual(payload["n_records"], 4)
        self.assertEqual(payload["n_groups"], 4)


class MlBenchmarkTests(unittest.TestCase):
    def test_record_table_excludes_experiment_and_crit_gen_leakage(self):
        lf = pd.DataFrame({"feature": [1.0, 2.0]}, index=["s1", "s2"])
        tsa = pd.DataFrame(
            {
                "state": ["s1", "s2"],
                "experiment": [0, 1],
                "Crit_gen": ["g1", "g2"],
                "Location": ["l1", "l2"],
                "Terminal": ["t1", "t2"],
                "Type": [0, 1],
                "CCT": [1.0, 2.0],
            }
        )

        X, _, _ = ml_benchmark.build_record_table(lf, tsa, scaler=_IdentityScaler())

        self.assertNotIn("experiment", X.columns)
        # Crit_gen is the simulation outcome, not a pre-fault-available input - must not
        # leak into the regression features (see CONTINGENCY_CATEGORICAL_COLUMNS).
        self.assertNotIn("Crit_gen", X.columns)
        self.assertIn("Location", X.columns)
        self.assertIn("Terminal", X.columns)
        self.assertIn("Type", X.columns)

    def test_record_table_can_match_retrieval_location_only_input_budget(self):
        lf = pd.DataFrame({"feature": [1.0, 2.0]}, index=["s1", "s2"])
        tsa = pd.DataFrame(
            {
                "state": ["s1", "s2"],
                "experiment": [0, 1],
                "Crit_gen": ["g1", "g2"],
                "Location": ["l1", "l2"],
                "Terminal": ["t1", "t2"],
                "Type": [0, 1],
                "CCT": [1.0, 2.0],
            }
        )

        X, _, _ = ml_benchmark.build_record_table(
            lf,
            tsa,
            scaler=_IdentityScaler(),
            contingency_cols=("Location",),
        )
        self.assertEqual(list(X.columns), ["feature", "Location"])

    def test_group_cv_returns_one_keyed_prediction_per_record(self):
        X = pd.DataFrame({"feature": range(6), "Location": ["l1", "l2"] * 3})
        y = pd.Series([1.0, 1.1, 2.0, 2.1, 3.0, 3.1])
        groups = pd.Series(["s1", "s1", "s2", "s2", "s3", "s3"])

        results, predictions = ml_benchmark.run_group_cv(
            X,
            y,
            groups,
            n_splits=3,
            model_names=set(),
        )

        self.assertEqual(set(results["model"]), {"location_median"})
        self.assertEqual(len(predictions), len(X))
        ordered = predictions.sort_values("record_position")
        self.assertEqual(ordered["cct_true"].tolist(), y.tolist())
        self.assertEqual(ordered["record_ordinal"].tolist(), [0, 1, 0, 1, 0, 1])


if __name__ == "__main__":
    unittest.main()
