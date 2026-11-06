import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tuned_comparators import check_folds, grid, make, select, tune_one, variants  # noqa: E402


class GridTests(unittest.TestCase):
    def test_fallback_drops_the_largest_budgets_first(self):
        budgets = lambda n: sorted({c["max_features"] for c in grid("extra_trees", n)}, key=str)  # noqa: E731
        self.assertEqual(len(grid("extra_trees")), 8)
        self.assertEqual(budgets(1), sorted({"sqrt", 0.1, 0.3}, key=str))
        self.assertEqual(budgets(3), ["sqrt"])
        with self.assertRaises(ValueError):
            grid("random_forest", 4)

    def test_bus39_adds_the_full_tree_grid_only_when_the_fallback_reduced_it(self):
        self.assertEqual([sfx for sfx, _ in variants("BUS39", 2)], ["", "_fullgrid"])
        self.assertEqual(variants("BUS39", 2)[1][1], ("extra_trees", "random_forest"))
        self.assertEqual(len(variants("BUS39", 0)), 1)
        self.assertEqual(len(variants("ELES", 2)), 1)

    def test_boosting_grid_ignores_the_fallback(self):
        self.assertEqual(grid("hist_gradient_boosting", 3), grid("hist_gradient_boosting"))
        self.assertEqual(len(grid("hist_gradient_boosting")), 9)


class EarlyStoppingTests(unittest.TestCase):
    def test_the_refit_reproduces_the_early_stopped_fit_it_was_scored_on(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(400, 5))
        y = X[:, 0] + rng.normal(scale=2.0, size=400)  # noisy, so validation loss stops improving early
        config, score = tune_one(
            "hist_gradient_boosting",
            {"learning_rate": 0.2, "max_leaf_nodes": 31},
            X[:300],
            y[:300],
            X[300:],
            y[300:],
            n_jobs=1,
        )
        self.assertLess(config["max_iter"], 1000)
        refit = make("hist_gradient_boosting", config, n_estimators=0, n_jobs=1).fit(X[:300], y[:300])
        self.assertAlmostEqual(float(np.mean(np.abs(refit.predict(X[300:]) - y[300:]))), score, places=12)


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
