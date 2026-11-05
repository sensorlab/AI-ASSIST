import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
from eles_exclusion_bounds import bounds  # noqa: E402


class BoundsTests(unittest.TestCase):
    def test_excluded_states_are_uncovered_and_referred(self):
        states = pd.DataFrame(
            {
                "partial": [True, True, False, True],
                "complete": [True, False, False, True],
                "referred_200ms": [False, True, True, True],
                "abstention": [False, True, True, False],
            }
        )
        rows = {r["quantity"]: r for r in bounds(states, n_attempted=8)}
        self.assertEqual(rows["partial"]["observed_rate"], 0.75)
        self.assertEqual(rows["partial"]["all_uncovered"], 3 / 8)
        self.assertEqual(rows["complete"]["all_uncovered"], 2 / 8)
        self.assertEqual(rows["referred_200ms"]["all_uncovered"], 7 / 8)
        self.assertEqual(rows["abstention"]["all_uncovered"], 6 / 8)

    def test_rejects_more_analysed_than_attempted(self):
        with self.assertRaises(ValueError):
            bounds(pd.DataFrame({"partial": [True, True]}), n_attempted=1)


if __name__ == "__main__":
    unittest.main()
