import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from _representations import build, family_mask  # noqa: E402
from eles_family_shapley import COALITIONS, PLAYERS, run_name, shapley, statistics  # noqa: E402


class ShapleyTests(unittest.TestCase):
    def test_additive_game_returns_contributions(self):
        weights = {"a": 1.0, "b": 2.5, "c": -0.5}
        phi = shapley(lambda s: sum(weights[p] for p in s), list(weights))
        for p, w in weights.items():
            self.assertAlmostEqual(phi[p], w)

    def test_efficiency_and_symmetry(self):
        rng = np.random.default_rng(0)
        table = {}

        def v(s):
            if not s:
                return 0.0
            return table.setdefault(s, float(rng.normal()))

        phi = shapley(v, ["a", "b", "c", "d"])
        self.assertAlmostEqual(sum(phi.values()), v(frozenset("abcd")))
        sym = shapley(lambda s: float(len(s & {"a", "b"}) > 0), ["a", "b", "c"])
        self.assertAlmostEqual(sym["a"], 0.5)
        self.assertAlmostEqual(sym["c"], 0.0)

    def test_run_names_reuse_existing_runs(self):
        names = [run_name(c) for c in COALITIONS]
        self.assertEqual(len(set(names)), 32)
        self.assertEqual(names[0], "uniform")
        self.assertEqual(names[-1], "exact_lines")
        self.assertEqual(run_name(frozenset({"angle"})), "only_angle")
        self.assertEqual(run_name(frozenset(PLAYERS) - {"voltage"}), "without_voltage")
        self.assertEqual(run_name(frozenset({"active", "voltage"})), "families_voltage+active")

    def test_statistics_sum_to_full_value(self):
        rng = np.random.default_rng(1)
        counts = rng.integers(1, 5, size=10).astype(float)
        errors = rng.random((10, 32)) * counts[:, None]
        s = statistics(counts, errors, np.ones(10))
        self.assertAlmostEqual(sum(s[f"shapley_{p}"] for p in PLAYERS), s["value_full"])
        self.assertAlmostEqual(sum(s[f"share_{p}"] for p in PLAYERS), 1.0)

    def test_coalition_mask_is_union(self):
        embed = ["U_a", "phi_a_cos", "phi_a_sin", "P_x", "Sk_a"]
        vectors = np.random.default_rng(2).normal(size=(4, 5))
        keys = np.array([[1], [1], [0], [0]], dtype=bool)
        _, facts = build("families_voltage+angle", vectors, embed, [], keys)
        union = family_mask(embed, "voltage") | family_mask(embed, "angle")
        self.assertEqual(facts["n_dimensions"], int(union.sum()))


if __name__ == "__main__":
    unittest.main()
