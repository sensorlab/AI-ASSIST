import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
from eles_compatibility_frontier import by_system, line_system  # noqa: E402


class LineSystemTests(unittest.TestCase):
    def test_coded_line_takes_its_prefix(self):
        self.assertEqual(line_system("oserv_LneOAAA-OBBB"), ("Austria", False))

    def test_local_first_node_falls_back_to_the_coded_second_node(self):
        self.assertEqual(line_system("oserv_LneRLOCAL-IBBB"), ("Italy", True))
        self.assertEqual(line_system("oserv_LneKLOCAL-OBBB-X"), ("Austria", False))

    def test_tie_line_is_flagged(self):
        self.assertEqual(line_system("oserv_LneQAAA-MBBB"), ("Hungary", True))

    def test_uncoded_line_is_unassigned(self):
        self.assertEqual(line_system("oserv_LneNAAA-ABBB"), ("unassigned", False))


class BySystemTests(unittest.TestCase):
    def test_aggregates_without_line_names(self):
        flags = pd.DataFrame(
            {
                "rule": "r",
                "availability": "a",
                "flag": ["f1", "f2", "f3"],
                "system": ["Italy", "Italy", "Austria"],
                "end_nodes_disagree": [True, False, False],
                "n_retrieved_mismatch": [6, 2, 2],
                "n_retrieved_neighbours": 20,
                "share_of_retrieved_neighbours": [0.3, 0.1, 0.1],
            }
        )
        out = by_system(flags)
        self.assertNotIn("flag", out.columns)
        italy = out.set_index("system").loc["Italy"]
        self.assertEqual((italy["n_lines_differing"], italy["n_lines_end_nodes_disagree"]), (2, 1))
        self.assertAlmostEqual(italy["share_of_mismatches"], 0.8)
        self.assertAlmostEqual(italy["max_line_share_of_neighbours"], 0.3)
        self.assertEqual(list(out["system"]), ["Italy", "Austria"])


if __name__ == "__main__":
    unittest.main()
