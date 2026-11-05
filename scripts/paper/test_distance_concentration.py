import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from distance_concentration import contrasts, permute_columns, relative_contrast  # noqa: E402


class RelativeContrastTests(unittest.TestCase):
    def test_value(self):
        self.assertEqual(relative_contrast(np.array([2.0, 3.0, 6.0])), 2.0)

    def test_undefined_and_infinite(self):
        self.assertTrue(np.isnan(relative_contrast(np.array([1.0]))))
        self.assertTrue(np.isinf(relative_contrast(np.array([0.0, 1.0]))))

    def test_query_excluded_and_groups_respected(self):
        x = np.array([[0.0], [1.0], [3.0], [10.0]])
        # Query 0 against all others: d = 1, 3, 10 -> (10 - 1) / 1.
        self.assertAlmostEqual(contrasts(x)[0], 9.0)
        # Within group {0, 1, 2}: d = 1, 3 -> 2; state 3 alone in its group -> undefined.
        rc = contrasts(x, groups=np.array([0, 0, 0, 1]))
        self.assertAlmostEqual(rc[0], 2.0)
        self.assertTrue(np.isnan(rc[3]))


class PermuteTests(unittest.TestCase):
    def test_keeps_each_column_marginal(self):
        x = np.arange(20.0).reshape(10, 2)
        p = permute_columns(x, np.random.default_rng(42))
        for j in range(2):
            np.testing.assert_array_equal(np.sort(p[:, j]), x[:, j])


if __name__ == "__main__":
    unittest.main()
