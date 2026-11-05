import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
from eles_uniform_weights import decision  # noqa: E402


def _rows(est, lo, hi):
    return pd.DataFrame(
        {
            "resampling_unit": ["state", "topology_group", "calendar_week"],
            "decides": [True, True, False],
            "estimate": [est] * 3,
            "ci_low": [lo, lo - 0.001, lo - 0.01],
            "ci_high": [hi, hi + 0.001, hi + 0.01],
        }
    )


class DecisionTests(unittest.TestCase):
    def test_widest_deciding_unit_is_used(self):
        self.assertEqual(decision(_rows(0.02, 0.015, 0.025))["resampling_unit"], "topology_group")

    def test_kernel_improves_needs_label_resolution(self):
        self.assertEqual(decision(_rows(0.012, 0.005, 0.02))["verdict"], "kernel_improves")
        self.assertEqual(decision(_rows(0.004, 0.002, 0.006))["verdict"], "negligible")

    def test_interval_spanning_zero(self):
        self.assertEqual(decision(_rows(0.012, 0.0, 0.03))["verdict"], "no_difference")

    def test_uniform_better_is_not_kernel_improves(self):
        self.assertEqual(decision(_rows(-0.004, -0.006, -0.002))["verdict"], "negligible")
        self.assertEqual(decision(_rows(-0.02, -0.03, -0.012))["verdict"], "uniform_improves")


if __name__ == "__main__":
    unittest.main()
