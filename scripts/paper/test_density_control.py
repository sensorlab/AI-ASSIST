import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from density_control import exclusions, partition_exact, partition_sampled, reading, reproduced_share  # noqa: E402

STATES = np.array([f"s{i}" for i in range(10)])


class PartitionTests(unittest.TestCase):
    def test_exact_partition_reproduces_the_size_multiset(self):
        sizes = np.array([1, 1, 3, 5])
        group_of = partition_exact(STATES, sizes, np.random.default_rng(0))
        self.assertEqual(set(group_of), set(STATES))
        self.assertEqual(sorted(Counter(group_of.values()).values()), [1, 1, 3, 5])

    def test_exact_partition_rejects_sizes_that_do_not_sum(self):
        with self.assertRaises(ValueError):
            partition_exact(STATES, np.array([3, 3]), np.random.default_rng(0))

    def test_sampled_partition_places_every_state_once_with_pool_sizes(self):
        group_of = partition_sampled(STATES, np.array([4]), np.random.default_rng(0))
        self.assertEqual(set(group_of), set(STATES))
        self.assertEqual(sorted(Counter(group_of.values()).values()), [2, 4, 4])  # last group truncated


class ExclusionTests(unittest.TestCase):
    def test_query_sees_only_its_own_group_in_its_archive(self):
        group_of = {"s0": 0, "s1": 0, "s2": 1}  # s3..s9 are outside the archive
        excl = exclusions(STATES, group_of)
        self.assertEqual(set(excl), {"s0", "s1", "s2"})
        self.assertEqual(set(STATES) - set(excl["s0"]), {"s1"})
        self.assertEqual(set(STATES) - set(excl["s2"]), set())
        self.assertIn("s0", excl["s0"])


class ShareTests(unittest.TestCase):
    def test_reproduced_share_and_reading(self):
        self.assertAlmostEqual(reproduced_share(arm=0.6, a0=1.0, eles=0.6), 1.0)
        self.assertAlmostEqual(reproduced_share(arm=0.9, a0=1.0, eles=0.6), 0.25)
        self.assertEqual(reading(0.8), "largely")
        self.assertEqual(reading(0.5), "partly")
        self.assertEqual(reading(0.1), "little")
        self.assertEqual(reading(-0.2), "little")
        self.assertEqual(reading(1.3), "overshoots")


if __name__ == "__main__":
    unittest.main()
