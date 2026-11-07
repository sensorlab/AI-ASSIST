import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
from chronological_evaluation import ORIGINS, blocks, coverage_loss  # noqa: E402


class BlockTests(unittest.TestCase):
    def setUp(self):
        times = pd.to_datetime(
            [
                "2024-06-30 10:00",
                "2024-07-31 23:15",
                "2024-08-01 00:00",
                "2024-08-15 00:00",
                "2024-12-31 23:59",
                "2025-01-26 00:00",
            ]
        )
        self.ts = pd.Series(times, index=pd.Index([f"s{i}" for i in range(len(times))]))

    def test_six_monthly_origins(self):
        self.assertEqual(len(ORIGINS), 6)
        self.assertEqual(str(ORIGINS[0].date()), "2024-08-01")
        self.assertEqual(str(ORIGINS[-1].date()), "2025-01-01")

    def test_test_blocks_partition_the_post_july_states_once(self):
        tested = [s for _, _, test in blocks(self.ts) for s in test]
        self.assertEqual(sorted(tested), ["s2", "s3", "s4", "s5"])
        self.assertEqual(len(tested), len(set(tested)))

    def test_history_strictly_precedes_its_origin_and_includes_everything_before_it(self):
        for origin, history, test in blocks(self.ts):
            self.assertTrue((self.ts[history] < origin).all())
            self.assertEqual(set(history), set(self.ts.index[self.ts < origin]))
            self.assertTrue((self.ts[test] >= origin).all())

    def test_last_block_runs_to_the_end(self):
        self.assertEqual(blocks(self.ts)[-1][2], ["s5"])


class CoverageLossTests(unittest.TestCase):
    def setUp(self):
        self.m = pd.DataFrame(
            {
                "state": ["a", "a", "b", "c"],
                "covered": [True, True, False, False],
                "covered_global": [True, True, True, False],
            }
        )

    def test_counts_records_and_states_lost_with_a_query_error(self):
        errors = pd.DataFrame({"state": ["a", "b", "c"], "query_error": [None, "weights underflow", None]})
        self.assertEqual(coverage_loss(self.m, errors), (1, 1))

    def test_stops_when_a_lost_state_has_no_query_error(self):
        errors = pd.DataFrame({"state": ["a", "b", "c"], "query_error": [None, None, None]})
        with self.assertRaises(ValueError):
            coverage_loss(self.m, errors)

    def test_stops_when_r_cont_covers_a_record_r_global_does_not(self):
        m = self.m.assign(covered=[True, True, False, True])
        errors = pd.DataFrame({"state": ["a", "b", "c"], "query_error": [None, "x", None]})
        with self.assertRaises(ValueError):
            coverage_loss(m, errors)


if __name__ == "__main__":
    unittest.main()
