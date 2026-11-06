import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
from alpha_k_sensitivity import assert_folds_closed, common_support, select  # noqa: E402


def _records(rows):
    """rows: (K, alpha, state, ordinal, covered, true, pred)."""
    cols = ["K", "alpha", "state", "record_ordinal", "covered", "cct_true", "pred_selection"]
    return pd.DataFrame(rows, columns=cols)


class SelectionTests(unittest.TestCase):
    def test_common_support_drops_records_any_setting_misses(self):
        r = _records(
            [
                (25, 1.0, "a", 0, True, 0.2, 0.2),
                (25, 1.0, "a", 1, False, 0.2, None),
                (100, 1.0, "a", 0, True, 0.2, 0.3),
                (100, 1.0, "a", 1, True, 0.2, 0.2),
            ]
        )
        support = common_support(r)
        self.assertEqual(set(support["record_ordinal"]), {0})

    def test_abstaining_setting_does_not_win(self):
        # K=25 is perfect on the one record it covers but misses the other; the common support is record 0 only, where
        # K=100 errs by 0.1, so K=25 is chosen on that record alone, not credited for skipping record 1.
        r = _records(
            [
                (25, 1.0, "a", 0, True, 0.2, 0.2),
                (25, 1.0, "a", 1, False, 0.2, None),
                (100, 1.0, "a", 0, True, 0.2, 0.3),
                (100, 1.0, "a", 1, True, 0.2, 0.9),
            ]
        )
        k, alpha, crit = select(r)
        self.assertEqual((k, alpha), (25, 1.0))
        self.assertEqual(set(crit["n_records"]), {1})

    def test_ties_go_to_the_setting_nearest_deployed(self):
        same = [(k, a, "a", 0, True, 0.2, 0.25) for k, a in ((400, 1.0), (100, 2.0), (100, 1.0), (50, 0.0))]
        self.assertEqual(select(_records(same))[:2], (100, 1.0))
        no_deployed = [(k, a, "a", 0, True, 0.2, 0.25) for k, a in ((400, 1.0), (100, 2.0), (50, 0.0))]
        self.assertEqual(select(_records(no_deployed))[:2], (400, 1.0))  # alpha distance is compared before K
        only_far = [(k, a, "a", 0, True, 0.2, 0.25) for k, a in ((50, 0.0), (100, 4.0))]
        self.assertEqual(select(_records(only_far))[:2], (100, 4.0))  # alpha = 0 is farthest


class FoldClosureTests(unittest.TestCase):
    def setUp(self):
        self.group_of = pd.Series({"a": "g1", "b": "g1", "c": "g2"})
        self.states = pd.DataFrame({"state": ["a", "b", "c"], "n_retrieved_line_mismatch": [0, 0, 0]})

    def test_closed_folds_pass(self):
        assert_folds_closed(self.states, self.group_of, [frozenset({"a", "b"}), frozenset({"c"})])

    def test_group_split_across_folds_fails(self):
        with self.assertRaises(AssertionError):
            assert_folds_closed(self.states, self.group_of, [frozenset({"a"}), frozenset({"b", "c"})])

    def test_cross_topology_neighbour_fails(self):
        states = self.states.assign(n_retrieved_line_mismatch=[0, 1, 0])
        with self.assertRaises(AssertionError):
            assert_folds_closed(states, self.group_of, [frozenset({"a", "b"}), frozenset({"c"})])


if __name__ == "__main__":
    unittest.main()
