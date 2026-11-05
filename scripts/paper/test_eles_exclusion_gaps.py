import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from eles_exclusion_gaps import matched, pairs  # noqa: E402


class GapPairTests(unittest.TestCase):
    def setUp(self):
        # Slots 0 1 2 3 _ 5 6: one single-slot gap between slots 3 and 5 (span 2).
        self.hours = np.array([0, 1, 2, 3, 5, 6])
        self.flags = np.array([[0], [0], [0], [1], [0], [0]], bool)

    def test_gap_and_span_matched_controls(self):
        df = pairs(self.hours, self.flags)
        gap = df[df["kind"] == "gap"]
        self.assertEqual(list(zip(gap["a"], gap["b"], gap["span"], strict=True)), [(3, 4, 2)])
        ctrl = df[df["kind"] == "control"]
        # Span-2 pairs with the middle slot present: (0,2) and (1,3) only; none crosses the gap.
        self.assertEqual(sorted(zip(ctrl["a"], ctrl["b"], strict=True)), [(0, 2), (1, 3)])

    def test_matched_difference(self):
        out = matched(pairs(self.hours, self.flags))
        # Gap pair (slot 3 vs 5) differs; controls: (0,2) same, (1,3) differs -> expected 0.5.
        self.assertEqual(out["key_differs_gap"], 1.0)
        self.assertEqual(out["key_differs_matched_control"], 0.5)
        self.assertEqual(out["key_differs_difference"], 0.5)


if __name__ == "__main__":
    unittest.main()
