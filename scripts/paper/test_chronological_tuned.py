import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chronological_tuned import CONFIGS, check_configs, majority_config  # noqa: E402


class MajorityConfigTests(unittest.TestCase):
    def test_reproduces_the_prespecified_boosting_configuration_from_08s_choices(self):
        chosen = [
            {"learning_rate": 0.1, "max_leaf_nodes": 255, "max_iter": 1000},
            {"learning_rate": 0.1, "max_leaf_nodes": 255, "max_iter": 825},
            {"learning_rate": 0.1, "max_leaf_nodes": 255, "max_iter": 1000},
            {"learning_rate": 0.05, "max_leaf_nodes": 255, "max_iter": 1000},
            {"learning_rate": 0.1, "max_leaf_nodes": 255, "max_iter": 755},
        ]
        self.assertEqual(majority_config(chosen), CONFIGS["hist_gradient_boosting"])

    def test_a_tie_goes_to_the_first_listed_grid_value(self):
        chosen = [{"max_features": 0.3, "min_samples_leaf": 5}, {"max_features": 0.1, "min_samples_leaf": 1}]
        self.assertEqual(majority_config(chosen), {"max_features": 0.1, "min_samples_leaf": 1})

    def test_an_iteration_tie_goes_to_the_larger_count(self):
        self.assertEqual(majority_config([{"max_iter": 755}, {"max_iter": 1000}]), {"max_iter": 1000})

    def test_configurations_match_analysis_08s_results(self):
        check_configs()


if __name__ == "__main__":
    unittest.main()
