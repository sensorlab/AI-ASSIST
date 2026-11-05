import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from screening_reliability import (  # noqa: E402
    conformal_margin,
    cross_fit_margins,
    match_load,
    online_margins,
)


class ConformalMarginTests(unittest.TestCase):
    def test_rank_rule(self):
        # n = 99, alpha = 0.01: rank ceil(0.99 * 100) = 99, the largest score, despite float rounding.
        scores = np.arange(1.0, 100.0)
        self.assertEqual(conformal_margin(scores, 0.01), 99.0)
        # n = 19, alpha = 0.1: rank ceil(0.9 * 20) = 18.
        self.assertEqual(conformal_margin(np.arange(1.0, 20.0), 0.1), 18.0)

    def test_too_few_scores_give_infinite_margin(self):
        self.assertEqual(conformal_margin(np.arange(1.0, 99.0), 0.01), math.inf)
        self.assertEqual(conformal_margin(np.array([]), 0.05), math.inf)

    def test_negative_margin_is_kept(self):
        self.assertEqual(conformal_margin(-np.arange(1.0, 20.0), 0.1), -2.0)


class MondrianTests(unittest.TestCase):
    def test_classes_are_calibrated_separately(self):
        # Fold 0 is the test fold; fold 1 calibrates. Scaled class residuals are 2 * s, single class residuals 5.
        s = np.array([0.5, np.nan, 1.0, 2.0, np.nan, np.nan])
        r = np.array([0.0, 0.0, 2.0, 4.0, 5.0, 5.0])
        full = np.ones(6, bool)
        fold = np.array([0, 0, 1, 1, 1, 1])
        m2, m3 = cross_fit_margins(r, s, full, fold, alpha=0.4)
        # P2: rank ceil(0.6 * 5) = 3 of [2, 4, 5, 5] -> 5.
        self.assertEqual(m2[0], 5.0)
        # P3 scaled: r/s = [2, 2], rank ceil(0.6 * 3) = 2 -> 2, times s = 0.5.
        self.assertEqual(m3[0], 1.0)
        # P3 single: [5, 5], rank 2 -> 5.
        self.assertEqual(m3[1], 5.0)

    def test_uncovered_states_never_calibrate(self):
        s = np.full(4, np.nan)
        r = np.array([0.0, 100.0, 1.0, 1.0])
        full = np.array([True, False, True, True])
        m2, _ = cross_fit_margins(r, s, full, np.array([0, 1, 1, 1]), alpha=0.5)
        self.assertEqual(m2[0], 1.0)


class OnlineTests(unittest.TestCase):
    def test_strictly_earlier_states_only(self):
        r = np.array([1.0, 2.0, 3.0, 4.0])
        s = np.full(4, np.nan)
        full = np.ones(4, bool)
        ts = np.array([0, 1, 1, 2])
        m2, _ = online_margins(r, s, full, ts, alpha=0.5)
        # t = 0: no history. t = 1: only state 0, rank ceil(0.5 * 2) = 1 -> 1, same for both states at t = 1.
        # t = 2: [1, 2, 3], rank 2 -> 2.
        np.testing.assert_array_equal(m2, [math.inf, 1.0, 1.0, 2.0])


class MatchLoadTests(unittest.TestCase):
    def test_adds_highest_scores_up_to_target(self):
        base = np.array([True, False, False, False, False])
        score = np.array([0.0, 1.0, 5.0, 3.0, 2.0])
        np.testing.assert_array_equal(match_load(base, score, 3), [True, False, True, True, False])

    def test_tie_at_cut_is_left_out_whole(self):
        base = np.array([False, False, False, False])
        score = np.array([3.0, 2.0, 2.0, 1.0])
        np.testing.assert_array_equal(match_load(base, score, 2), [True, False, False, False])

    def test_never_removes_base_referrals(self):
        base = np.array([True, True, False])
        np.testing.assert_array_equal(match_load(base, np.array([0.0, 0.0, 9.0]), 1), base)

    def test_ignores_scores_of_referred_states(self):
        base = np.array([True, False, False])
        score = np.array([np.nan, 1.0, 2.0])
        np.testing.assert_array_equal(match_load(base, score, 2), [True, False, True])


if __name__ == "__main__":
    unittest.main()
