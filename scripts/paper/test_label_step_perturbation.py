import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
from label_step_perturbation import shifted  # noqa: E402


class ShiftTests(unittest.TestCase):
    def setUp(self):
        # Lowest CCT per state: a 0.18, b 0.19, c 0.20, d 0.21; every record covered, estimates 0.25 (never referred).
        self.records = pd.DataFrame(
            {
                "state": ["a", "a", "b", "c", "d"],
                "covered": True,
                "cct_true": [0.18, 0.30, 0.19, 0.20, 0.21],
                "pred_screening_min": 0.25,
            }
        )

    def test_referral_is_unchanged_by_any_shift(self):
        frames = [shifted(self.records, 0.20, s)["referred"] for s in (-1, 0, 1)]
        self.assertTrue(frames[0].equals(frames[1]) and frames[1].equals(frames[2]))

    def test_exactly_the_boundary_states_flip(self):
        unsafe = {s: shifted(self.records, 0.20, s)["unsafe"] for s in (-1, 0, 1)}
        self.assertEqual(list(unsafe[0]), [True, True, False, False])
        # -1 step: the state at exactly tau (c, 0.20) becomes unsafe; 0.19 + 0.01 compares equal to tau under +1.
        self.assertEqual(list(unsafe[-1]), [True, True, True, False])
        self.assertEqual(list(unsafe[1]), [True, False, False, False])
        self.assertEqual(list(shifted(self.records, 0.20, 0)["boundary"]), [False, True, True, False])


if __name__ == "__main__":
    unittest.main()
