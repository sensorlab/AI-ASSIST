import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dataset_facts import flag_changes, share_sharing_a_key  # noqa: E402


class FactTests(unittest.TestCase):
    def test_flag_changes_counts_flips_between_consecutive_states(self):
        bits = np.array([[0, 1], [1, 1], [1, 0], [1, 0]], dtype=bool)
        self.assertEqual(flag_changes(bits), 2)

    def test_share_counts_every_state_whose_key_recurs(self):
        self.assertAlmostEqual(share_sharing_a_key(pd.Series(["a", "b", "a", "c"])), 0.5)


if __name__ == "__main__":
    unittest.main()
