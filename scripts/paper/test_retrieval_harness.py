import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _retrieval_harness import BruteForceDatabase, Rule, causal_exclusions  # noqa: E402

from src.services.qdrant.repository import DatabaseQdrant  # noqa: E402

FLAGS = ["oserv_Lne_a", "oserv_Lne_b", "oserv_Lne_c"]


def _corpus(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    features = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"x{i}" for i in range(5)])
    # Three topology groups that differ in one or two flags, so exact and relaxed rules disagree.
    patterns = np.array([[1, 1, 1], [1, 1, 0], [0, 0, 1]], dtype=float)
    flags = pd.DataFrame(patterns[np.arange(n) % 3], columns=FLAGS)
    frame = pd.concat([features, flags], axis=1)
    frame.index = pd.Index([f"s{i}" for i in range(n)], name="state")
    return frame


class BruteForceDatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = _corpus()
        self.qdrant = DatabaseQdrant(
            client=QdrantClient(":memory:"),
            collection_name="harness_test",
            subset_topology_cols=FLAGS,
            use_population_lock=False,
        ).fit(self.frame)

    def _brute(self, rule: Rule) -> BruteForceDatabase:
        return BruteForceDatabase(self.frame, {"lines_only": FLAGS}, rule)

    def test_exact_rule_matches_qdrant_neighbours_and_vectors(self):
        brute = self._brute(Rule("exact", "lines_only", 0))
        for uid in ("s0", "s1", "s5", "s31"):
            query = self.frame.loc[[uid]]
            for limit in (3, 100):
                expected = self.qdrant.query(query, limit=limit, exclude_source_index=[uid])
                got = brute.query(query, limit=limit, exclude_source_index=[uid])
                self.assertEqual(list(got.rows.index), list(expected.rows.index))
                np.testing.assert_array_equal(got.rows.to_numpy(), expected.rows.to_numpy())
                self.assertEqual(list(got.rows.columns), list(expected.rows.columns))

    def test_exclusion_list_withholds_every_listed_state(self):
        brute = self._brute(Rule("exact", "lines_only", 0))
        withheld = [f"s{i}" for i in range(0, 60, 3)][1:]
        got = brute.query(self.frame.loc[["s0"]], exclude_source_index=["s0", *withheld])
        self.assertEqual(len(got.rows), 0)

    def test_hamming_rule_admits_one_flag_difference_only(self):
        brute = self._brute(Rule("h1", "lines_only", 1))
        got = brute.query(self.frame.loc[["s0"]], exclude_source_index=["s0"])
        groups = {int(uid[1:]) % 3 for uid in got.rows.index}
        self.assertEqual(groups, {0, 1})  # pattern 2 differs from pattern 0 in two flags

    def test_no_filter_returns_global_nearest_neighbours(self):
        brute = self._brute(Rule("none", None))
        got = brute.query(self.frame.loc[["s0"]], limit=5, exclude_source_index=["s0"])
        x = self.frame[[f"x{i}" for i in range(5)]].to_numpy(dtype=np.float32).astype(np.float64)
        d = np.sqrt(((x - x[0]) ** 2).sum(axis=1))
        d[0] = np.inf
        self.assertEqual(list(got.rows.index), [f"s{i}" for i in np.argsort(d, kind="stable")[:5]])


class CausalExclusionTests(unittest.TestCase):
    def test_lag_zero_keeps_earlier_and_simultaneous_states_but_never_the_query(self):
        ts = pd.Series(pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 01:00", "2024-01-01 02:00"]))
        ts.index = ["a", "b", "c", "d"]
        excl = causal_exclusions(ts, 0)
        self.assertEqual(sorted(excl["a"]), ["a", "b", "c", "d"])
        self.assertEqual(sorted(excl["b"]), ["b", "d"])
        self.assertEqual(sorted(excl["d"]), ["d"])

    def test_lag_withholds_states_younger_than_the_lag(self):
        ts = pd.Series(pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:30", "2024-01-01 01:00"]))
        ts.index = ["a", "b", "c"]
        self.assertEqual(sorted(causal_exclusions(ts, 1)["c"]), ["b", "c"])


if __name__ == "__main__":
    unittest.main()
