import math
import unittest
from unittest.mock import patch

import numpy as np

from src.domain.estimation.models import ReportSummary
from src.domain.estimation.service import (
    _distance_summary,
    _effective_sample_size,
    _neighborhood_compactness,
    _normalized_weights,
)
from src.domain.estimation.weights import K


class NeighborhoodQualityIndicatorTests(unittest.TestCase):
    def test_effective_sample_size_equals_n_for_uniform_weights(self):
        weights = np.full(4, 0.25)

        self.assertAlmostEqual(_effective_sample_size(weights), 4.0)

    def test_effective_sample_size_approaches_one_for_dominant_weight(self):
        weights = np.array([0.999, 0.0005, 0.0005])

        self.assertAlmostEqual(_effective_sample_size(weights), 1.002, places=3)

    def test_normalized_weights_are_uniform_for_equal_distances(self):
        weights = _normalized_weights(np.array([2.0, 2.0, 2.0]), crit_gen="G1")

        np.testing.assert_allclose(weights, np.full(3, 1.0 / 3.0))

    def test_neighborhood_compactness_excludes_diagonal_terms(self):
        X = np.array([[0.0], [1.0]])

        self.assertAlmostEqual(_neighborhood_compactness(X, crit_gen="G1"), math.exp(-1.0))

    def test_neighborhood_compactness_uses_unique_unordered_pairs_and_is_normalized(self):
        X = np.array([[0.0], [1.0], [2.0]])
        expected = float(np.mean(K(np.array([1.0, 2.0, 1.0]))))

        self.assertAlmostEqual(_neighborhood_compactness(X, crit_gen="G1"), expected)

    def test_neighborhood_compactness_does_not_double_count_symmetric_pairs(self):
        X = np.array([[0.0], [1.0], [2.0]])
        seen_distances: list[np.ndarray] = []

        def capture_kernel(distances: np.ndarray) -> np.ndarray:
            seen_distances.append(distances.copy())
            return K(distances)

        with patch("src.domain.estimation.service.K", side_effect=capture_kernel):
            _neighborhood_compactness(X, crit_gen="G1")

        self.assertEqual(len(seen_distances), 1)
        np.testing.assert_allclose(seen_distances[0], np.array([1.0, 2.0, 1.0]))

    def test_neighborhood_compactness_does_not_grow_with_duplicated_points(self):
        X = np.array([[0.0], [0.0], [1.0], [1.0]])
        expected = float(np.mean(K(np.array([0.0, 1.0, 1.0, 1.0, 1.0, 0.0]))))

        self.assertAlmostEqual(_neighborhood_compactness(X, crit_gen="G1"), expected)
        self.assertLessEqual(_neighborhood_compactness(X, crit_gen="G1"), 1.0)

    def test_neighborhood_compactness_is_normalized_across_group_sizes(self):
        small = np.zeros((2, 1))
        large = np.zeros((5, 1))

        self.assertAlmostEqual(_neighborhood_compactness(small, crit_gen="G1"), 1.0)
        self.assertAlmostEqual(_neighborhood_compactness(large, crit_gen="G1"), 1.0)

    def test_neighborhood_compactness_is_none_for_single_or_empty_group(self):
        self.assertIsNone(_neighborhood_compactness(np.empty((0, 2)), crit_gen="G1"))
        self.assertIsNone(_neighborhood_compactness(np.array([[1.0, 2.0]]), crit_gen="G1"))

    def test_distance_summary_uses_query_distances(self):
        qds = np.array([3.0, 1.0, 2.0])

        self.assertEqual(
            _distance_summary(qds),
            {
                "min": 1.0,
                "mean": 2.0,
                "median": 2.0,
                "spread": 2.0,
                "norm": 1.0 / (2.0 + 1e-12),
            },
        )

    def test_report_summary_accepts_neighborhood_compactness_values_and_none(self):
        base = {
            "cct_weighted": 1.0,
            "cct_weighted_per_location": {"L1": 1.0},
            "location_weight_mass": {"L1": 1.0},
            "n": 1,
            "n_eff": 1.0,
            "distances": {"min": 0.0, "mean": 0.0, "median": 0.0, "spread": 0.0, "norm": 0.0},
            "location_counts": {"L1": 1},
        }

        with_value = ReportSummary(**base, neighborhood_compactness=0.5)
        with_none = ReportSummary(**base, neighborhood_compactness=None)

        self.assertEqual(with_value.neighborhood_compactness, 0.5)
        self.assertIsNone(with_none.neighborhood_compactness)


if __name__ == "__main__":
    unittest.main()
