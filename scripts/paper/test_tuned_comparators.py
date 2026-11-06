import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tuned_comparators import check_folds, grid, select  # noqa: E402


class GridTests(unittest.TestCase):
    def test_fallback_drops_the_largest_budgets_first(self):
        budgets = lambda n: sorted({c["max_features"] for c in grid("extra_trees", n)}, key=str)  # noqa: E731
        self.assertEqual(len(grid("extra_trees")), 8)
        self.assertEqual(budgets(1), sorted({"sqrt", 0.1, 0.3}, key=str))
        self.assertEqual(budgets(3), ["sqrt"])
        with self.assertRaises(ValueError):
            grid("random_forest", 4)

    def test_boosting_grid_ignores_the_fallback(self):
        self.assertEqual(grid("hist_gradient_boosting", 3), grid("hist_gradient_boosting"))
        self.assertEqual(len(grid("hist_gradient_boosting")), 18)


class SelectTests(unittest.TestCase):
    def test_lowest_score_wins_and_ties_go_to_the_first_listed(self):
        self.assertEqual(select([0.3, 0.1, 0.2]), 1)
        self.assertEqual(select([0.2, 0.1, 0.1]), 1)


class FoldTests(unittest.TestCase):
    def setUp(self):
        self.groups = pd.Series(["a", "a", "b", "c"])
        self.folds = [(np.array([2, 3]), np.array([0, 1])), (np.array([0, 1]), np.array([2, 3]))]

    def test_matching_folds_pass(self):
        stored = pd.DataFrame({"state": ["a", "a", "b", "c"], "fold": [0, 0, 1, 1]})
        check_folds(self.groups, self.folds, stored)

    def test_a_state_in_another_fold_fails(self):
        stored = pd.DataFrame({"state": ["a", "b", "c"], "fold": [0, 0, 1]})
        with self.assertRaises(AssertionError):
            check_folds(self.groups, self.folds, stored)


if __name__ == "__main__":
    unittest.main()
