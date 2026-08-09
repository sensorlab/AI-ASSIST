import unittest

import numpy as np

from src.benchmarking import risk_coverage_point


class RiskCoveragePointTests(unittest.TestCase):
    def test_fractional_matches_hard_when_no_tie_straddles_a_cutoff(self):
        metric = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        err = np.array([9.0, 3.0, 7.0, 1.0, 5.0, 2.0, 8.0, 4.0, 6.0, 10.0])
        for higher_is_better in (False, True):
            hard = risk_coverage_point(metric, err, higher_is_better=higher_is_better, tie_policy="hard")
            frac = risk_coverage_point(metric, err, higher_is_better=higher_is_better, tie_policy="fractional")
            for cov in hard:
                self.assertAlmostEqual(hard[cov][0], frac[cov][0], places=12)

    def test_fractional_boundary_group_matches_hand_computed_expected_mean(self):
        # Group (metric=1): err {10, 20}; group (metric=2): err {30, 40, 50}. Lower is
        # better, so group 1 is fully trusted before group 2. At coverage giving k=3,
        # the cutoff falls one record into the second group.
        metric = np.array([1.0, 1.0, 2.0, 2.0, 2.0])
        err = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        n = len(err)
        k_target = 3
        coverage = k_target / n
        out = risk_coverage_point(metric, err, higher_is_better=False, tie_policy="fractional", coverages=(coverage,))
        # Expected MAE = mean(fully-included group) blended with the *expected* mean of a
        # uniform 1-of-3 draw from the boundary group, which by symmetry equals that group's
        # own mean: (10 + 20 + mean(30, 40, 50)) / 3.
        expected_mae = (10.0 + 20.0 + (30.0 + 40.0 + 50.0) / 3.0) / 3.0
        self.assertAlmostEqual(out[coverage][0], expected_mae, places=12)
        self.assertTrue(np.isnan(out[coverage][1]), "fractional policy must not report a RMSE value")

    def test_fractional_boundary_inside_first_group(self):
        # Single tied group of size 3; cutoff falls after 2 of its 3 members.
        metric = np.array([1.0, 1.0, 1.0, 2.0, 2.0])
        err = np.array([10.0, 20.0, 30.0, 100.0, 200.0])
        coverage = 2 / 5
        out = risk_coverage_point(metric, err, higher_is_better=False, tie_policy="fractional", coverages=(coverage,))
        # Expected mean of a uniform 2-of-3 draw from {10, 20, 30} equals the group mean.
        expected_mae = (10.0 + 20.0 + 30.0) / 3.0
        self.assertAlmostEqual(out[coverage][0], expected_mae, places=12)

    def test_higher_is_better_direction(self):
        metric = np.array([1.0, 2.0, 3.0, 4.0])
        err = np.array([100.0, 200.0, 300.0, 400.0])
        # Lower-is-better: most trusted are the smallest metric values (err 100, 200, ...).
        lower = risk_coverage_point(metric, err, higher_is_better=False, tie_policy="hard", coverages=(0.5,))
        self.assertAlmostEqual(lower[0.5][0], 150.0, places=12)
        # Higher-is-better: most trusted are the largest metric values (err 400, 300, ...).
        higher = risk_coverage_point(metric, err, higher_is_better=True, tie_policy="hard", coverages=(0.5,))
        self.assertAlmostEqual(higher[0.5][0], 350.0, places=12)

    def test_fractional_is_invariant_to_row_permutation(self):
        rng = np.random.default_rng(0)
        # Deliberately heavy ties (only 3 distinct metric values across 30 records) so a
        # coverage cutoff is very likely to land inside a tied group.
        metric = rng.choice([1.0, 2.0, 3.0], size=30)
        err = rng.random(30) * 10.0
        baseline = risk_coverage_point(metric, err, higher_is_better=False, tie_policy="fractional")
        for _ in range(5):
            perm = rng.permutation(len(metric))
            permuted = risk_coverage_point(metric[perm], err[perm], higher_is_better=False, tie_policy="fractional")
            for cov in baseline:
                self.assertAlmostEqual(baseline[cov][0], permuted[cov][0], places=12)

    def test_hard_policy_can_depend_on_row_order_when_a_tie_straddles_a_cutoff(self):
        # Documents the exact problem the fractional policy fixes: with a coverage cutoff
        # strictly inside a tied group, argsort's tie order is a function of input position,
        # so "hard" truncation is not guaranteed permutation-invariant. This is not a
        # property being asserted as *desirable* - it is why "fractional" exists.
        metric = np.array([1.0] * 6)
        err = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        coverage = 3 / 6
        forward = risk_coverage_point(metric, err, higher_is_better=False, tie_policy="hard", coverages=(coverage,))
        reversed_err = err[::-1]
        backward = risk_coverage_point(
            metric, reversed_err, higher_is_better=False, tie_policy="hard", coverages=(coverage,)
        )
        # Both are valid "hard" outcomes on the same multiset of tied records but pick a
        # different 3-of-6 subset (first three vs. last three in traversal order), and the
        # fractional policy is exactly the fix that removes this order-dependence.
        self.assertNotAlmostEqual(forward[coverage][0], backward[coverage][0])
        fractional_forward = risk_coverage_point(
            metric, err, higher_is_better=False, tie_policy="fractional", coverages=(coverage,)
        )
        fractional_backward = risk_coverage_point(
            metric, reversed_err, higher_is_better=False, tie_policy="fractional", coverages=(coverage,)
        )
        self.assertAlmostEqual(fractional_forward[coverage][0], fractional_backward[coverage][0], places=12)

    def test_duplicate_rows_from_bootstrap_resampling_do_not_crash(self):
        metric = np.array([1.0, 1.0, 2.0, 3.0])
        err = np.array([10.0, 10.0, 20.0, 30.0])
        # Simulate a with-replacement bootstrap draw that duplicates a row multiple times.
        idx = np.array([0, 0, 0, 1, 2, 3, 3])
        boot_metric, boot_err = metric[idx], err[idx]
        for policy in ("hard", "fractional"):
            out = risk_coverage_point(boot_metric, boot_err, higher_is_better=False, tie_policy=policy)
            for mae, _rmse in out.values():
                self.assertTrue(np.isfinite(mae))

    def test_all_tie_policies_agree_at_full_coverage(self):
        rng = np.random.default_rng(1)
        metric = rng.choice([1.0, 2.0, 3.0, 4.0], size=50)
        err = rng.random(50) * 5.0
        expected_mean = float(err.mean())
        hard = risk_coverage_point(metric, err, higher_is_better=False, tie_policy="hard", coverages=(1.0,))
        frac = risk_coverage_point(metric, err, higher_is_better=False, tie_policy="fractional", coverages=(1.0,))
        randomized = risk_coverage_point(
            metric, err, higher_is_better=False, tie_policy="randomized", coverages=(1.0,), rng=rng
        )
        self.assertAlmostEqual(hard[1.0][0], expected_mean, places=12)
        self.assertAlmostEqual(frac[1.0][0], expected_mean, places=12)
        self.assertAlmostEqual(randomized[1.0][0], expected_mean, places=12)

    def test_randomized_requires_rng(self):
        metric = np.array([1.0, 2.0, 3.0])
        err = np.array([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            risk_coverage_point(metric, err, higher_is_better=False, tie_policy="randomized")

    def test_unknown_tie_policy_raises(self):
        metric = np.array([1.0, 2.0, 3.0])
        err = np.array([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            risk_coverage_point(metric, err, higher_is_better=False, tie_policy="bogus")


if __name__ == "__main__":
    unittest.main()
