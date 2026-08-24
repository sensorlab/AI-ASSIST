import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eles_chronological_topology_support import _has_earlier_same_topology  # noqa: E402


class HasEarlierSameTopologyTests(unittest.TestCase):
    def test_tied_minimum_timestamps_are_not_marked_as_having_an_earlier_state(self):
        # Group "a": two rows tied at the group minimum, one strictly later.
        # Group "b": a single row, trivially no earlier state.
        df = pd.DataFrame(
            {
                "group": ["a", "a", "a", "b"],
                "dt": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-01"]),
            }
        )
        result = _has_earlier_same_topology(df)
        self.assertEqual(result.tolist(), [False, False, True, False])

    def test_no_ties_matches_first_in_group_intuition(self):
        df = pd.DataFrame(
            {
                "group": ["a", "a", "a"],
                "dt": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            }
        )
        result = _has_earlier_same_topology(df)
        self.assertEqual(result.tolist(), [False, True, True])

    def test_lag_excludes_a_predecessor_within_the_lag_window(self):
        # Predecessor is only 10 minutes earlier; a 20-minute lag means it is not yet available.
        df = pd.DataFrame(
            {
                "group": ["a", "a"],
                "dt": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:10"]),
            }
        )
        no_lag = _has_earlier_same_topology(df)
        lagged = _has_earlier_same_topology(df, lag=pd.Timedelta(minutes=20))
        self.assertEqual(no_lag.tolist(), [False, True])
        self.assertEqual(lagged.tolist(), [False, False])

    def test_lag_does_not_exclude_a_predecessor_outside_the_lag_window(self):
        df = pd.DataFrame(
            {
                "group": ["a", "a"],
                "dt": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00"]),
            }
        )
        lagged = _has_earlier_same_topology(df, lag=pd.Timedelta(minutes=20))
        self.assertEqual(lagged.tolist(), [False, True])

    def test_lag_is_inclusive_at_exact_completion(self):
        # Predecessor completes exactly 20 minutes before the query - availability is inclusive
        # at exact completion (candidate_dt + lag <= query_dt), not strictly after (Codex
        # review, ai2ai.md, 2026-08-10 - the earlier strict-only version wrongly excluded this).
        df = pd.DataFrame(
            {
                "group": ["a", "a"],
                "dt": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:20"]),
            }
        )
        lagged = _has_earlier_same_topology(df, lag=pd.Timedelta(minutes=20))
        self.assertEqual(lagged.tolist(), [False, True])


if __name__ == "__main__":
    unittest.main()
