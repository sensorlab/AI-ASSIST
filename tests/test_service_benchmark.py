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
                    "summary": {
                        "cct_weighted": 2.0,
                        "cct_weighted_per_location": {"l1": 1.5},
                        "location_weight_mass": {"l1": 1.0},
                        "neighborhood_compactness": 1.0,
                        "n": 1,
                        "n_eff": 1.0,
                        "distances": {
                            "min": 0.1,
                            "mean": 0.1,
                            "median": 0.1,
                            "spread": 0.0,
                            "norm": 1.0,
                        },
                        "location_counts": {"l1": 1},
                    },
                    "included_state_ids": included_state_ids or [],
                    "per_neighbor": [],
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

        self.assertEqual(requests[0]["endpoint"], "http://localhost:8000/api/v1/estimate/by-generator")
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
        results = payload["results"].set_index("fold")

        self.assertEqual(results.loc[0, "coverage"], 1.0)
        self.assertEqual(results.loc[0, "mae"], 0.0)
        self.assertEqual(results.loc[1, "coverage"], 0.5)
        self.assertEqual(results.loc[1, "mae"], 0.0)
        self.assertEqual(payload["n_records"], 4)
        self.assertEqual(payload["n_groups"], 4)


class MlBenchmarkTests(unittest.TestCase):
    def test_record_table_excludes_experiment_and_includes_known_crit_gen(self):
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
        self.assertIn("Crit_gen", X.columns)


if __name__ == "__main__":
    unittest.main()
