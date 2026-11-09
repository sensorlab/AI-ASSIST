import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from _representations import (  # noqa: E402
    FAMILIES,
    bandwidth_constant,
    build,
    embed_mask,
    family_mask,
    same_key_pairs,
    slovenian_raw_columns,
)


class RepresentationTests(unittest.TestCase):
    def test_bandwidth_constant_matches_medians(self):
        ref, cand = np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0])
        c = bandwidth_constant(ref, cand)
        self.assertAlmostEqual(np.median(c * cand), np.median(ref))

    def test_angle_pairs_stay_together(self):
        embed = ["U_BusA_[pu]", "phi_BusA_[deg]_cos", "phi_BusA_[deg]_sin", "U_BusB_[pu]", "phi_BusB_[deg]_cos"]
        mask = embed_mask(embed, ["U_BusA_[pu]", "phi_BusA_[deg]"])
        self.assertEqual(mask.tolist(), [True, True, True, False, False])

    def test_slovenian_selection_strips_type_prefix(self):
        cols = ["U_BusALPHA_[pu]", "P_LodX1_[MW]", "Sk_ALPHA_[MVA]", "Q_GenOTHER_[MVAr]", "P1_LneL7_[MW].1"]
        self.assertEqual(
            slovenian_raw_columns(cols, {"ALPHA", "X1", "L7"}),
            ["U_BusALPHA_[pu]", "P_LodX1_[MW]", "Sk_ALPHA_[MVA]", "P1_LneL7_[MW].1"],
        )

    def test_same_key_pairs(self):
        keys = np.array([[1, 0], [1, 0], [0, 1], [1, 0]], dtype=bool)
        a, b = same_key_pairs(keys)
        self.assertEqual(sorted(zip(a.tolist(), b.tolist(), strict=True)), [(0, 1), (0, 3), (1, 3)])

    def test_l1_scaled_to_euclidean_median(self):
        rng = np.random.default_rng(0)
        vectors = rng.normal(size=(6, 4))
        keys = np.array([[1], [1], [1], [0], [0], [0]], dtype=bool)
        dist, facts = build("l1", vectors, [f"c{i}" for i in range(4)], [], keys)
        a, b = same_key_pairs(keys)
        l1 = np.abs(vectors[a] - vectors[b]).sum(axis=1)
        self.assertTrue(np.allclose(dist(vectors[a], vectors[b]), facts["bandwidth_constant"] * l1))
        self.assertAlmostEqual(
            np.median(dist(vectors[a], vectors[b])), np.median(np.linalg.norm(vectors[a] - vectors[b], axis=1))
        )

    def test_families_partition_the_embedding(self):
        embed = [
            "U_BusA_[pu]",
            "phi_BusA_[deg]_cos",
            "phi_BusA_[deg]_sin",
            "P_LodX_[MW]",
            "P1_LneL_[MW]",
            "Q2_LneL_[MVAr]",
            "Sk_BusA_[MVA]",
        ]
        masks = np.array([family_mask(embed, f) for f in FAMILIES])
        self.assertTrue((masks.sum(axis=0) == 1).all())
        self.assertEqual(family_mask(embed, "angle").tolist(), [False, True, True, False, False, False, False])
        self.assertEqual(family_mask(embed, "active").tolist(), [False, False, False, True, True, False, False])

    def test_unknown_family_is_an_error(self):
        with self.assertRaises(ValueError):
            family_mask(["U_BusA_[pu]", "X_BusA_[pu]"], "voltage")

    def test_without_family_drops_its_columns(self):
        rng = np.random.default_rng(1)
        vectors = rng.normal(size=(4, 3))
        embed, keys = ["U_a", "phi_a_cos", "Sk_a"], np.array([[1], [1], [0], [0]], dtype=bool)
        dist, facts = build("without_angle", vectors, embed, [], keys)
        self.assertEqual(facts["n_dimensions"], 2)
        expected = facts["bandwidth_constant"] * np.linalg.norm(
            vectors[[1]][:, [0, 2]] - vectors[[0]][:, [0, 2]], axis=1
        )
        self.assertTrue(np.allclose(dist(vectors[[0]], vectors[[1]]), expected))


if __name__ == "__main__":
    unittest.main()
