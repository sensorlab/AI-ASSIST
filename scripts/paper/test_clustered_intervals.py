import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from referral_decomposition import _bootstrap, _mean  # noqa: E402


class ClusterBootstrapTests(unittest.TestCase):
    def test_whole_clusters_are_drawn(self):
        # Cluster 0: three states valued 0; cluster 1: one state valued 1. Drawing two whole clusters can only give
        # a mean of 0, 1/4 or 1; resampling states would also give 1/2, 3/4 and others.
        arrays = {"x": np.array([0.0, 0.0, 0.0, 1.0])}
        clusters = np.array([0, 0, 0, 1])
        ci = _bootstrap(arrays, {"m": _mean("x")}, clusters, np.random.default_rng(1), n_resamples=200)
        self.assertIn(ci["m"][0], (0.0, 0.25, 1.0))
        self.assertIn(ci["m"][1], (0.0, 0.25, 1.0))

    def test_resample_count_is_honoured(self):
        arrays = {"x": np.arange(10.0)}
        a = _bootstrap(arrays, {"m": _mean("x")}, None, np.random.default_rng(0), n_resamples=5)
        b = _bootstrap(arrays, {"m": _mean("x")}, None, np.random.default_rng(0), n_resamples=5000)
        self.assertNotEqual(a, b)


if __name__ == "__main__":
    unittest.main()
