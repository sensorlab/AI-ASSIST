import json
import tempfile
import unittest
from pathlib import Path

from src.domain.estimation.service import _dataset_paths


class DatasetPathsTopologyVariantTests(unittest.TestCase):
    """_dataset_paths()'s topology_cols resolution: variant-specific file preferred when
    present, falling back to the single unversioned topology_cols.json for datasets that
    only ever had one definition (bus39, interscada/*, eles/2026-01) - see
    datasets/eles/2026-06/README.md's "Topology Variants" section for why this exists."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_dir = Path(self._tmp.name)

    def _make_dataset(self, name: str, *, topology_files: dict[str, list[str]]) -> None:
        interim = self.data_dir / name / "interim"
        processed = self.data_dir / name / "processed"
        interim.mkdir(parents=True)
        processed.mkdir(parents=True)
        (interim / "lf.pkl").write_bytes(b"")
        (processed / "tsa.db").write_bytes(b"")
        for filename, cols in topology_files.items():
            (processed / filename).write_text(json.dumps(cols))

    def test_multi_variant_dataset_resolves_requested_variant(self):
        self._make_dataset(
            "eles/2026-06",
            topology_files={
                "topology_cols_full.json": ["oserv_Gen1", "oserv_Lne1"],
                "topology_cols_lines_only.json": ["oserv_Lne1"],
                "topology_cols_slovenia_only.json": ["oserv_Lne1"],
            },
        )

        _, _, path_full = _dataset_paths(self.data_dir, "eles/2026-06", topology_variant="full")
        _, _, path_lines = _dataset_paths(self.data_dir, "eles/2026-06", topology_variant="lines_only")
        _, _, path_slovenia = _dataset_paths(self.data_dir, "eles/2026-06", topology_variant="slovenia_only")

        self.assertEqual(path_full.name, "topology_cols_full.json")
        self.assertEqual(path_lines.name, "topology_cols_lines_only.json")
        self.assertEqual(path_slovenia.name, "topology_cols_slovenia_only.json")

    def test_multi_variant_dataset_falls_back_to_bare_name_when_variant_is_none(self):
        self._make_dataset(
            "eles/2026-06",
            topology_files={
                "topology_cols.json": ["oserv_Gen1", "oserv_Lne1"],
                "topology_cols_lines_only.json": ["oserv_Lne1"],
            },
        )

        _, _, path = _dataset_paths(self.data_dir, "eles/2026-06", topology_variant=None)

        self.assertEqual(path.name, "topology_cols.json")

    def test_single_variant_dataset_ignores_requested_variant_and_uses_bare_file(self):
        """bus39/interscada/eles-2026-01 only ever wrote one topology_cols.json - requesting
        any variant name for them must not raise, and must resolve to that same bare file."""
        self._make_dataset("bus39", topology_files={"topology_cols.json": ["oserv_a", "oserv_b"]})

        for variant in ("full", "lines_only", "slovenia_only", None):
            _, _, path = _dataset_paths(self.data_dir, "bus39", topology_variant=variant)
            self.assertEqual(path.name, "topology_cols.json", f"variant={variant!r}")

    def test_missing_topology_cols_still_raises_when_no_variant_matches_and_no_bare_file(self):
        interim = self.data_dir / "broken" / "interim"
        processed = self.data_dir / "broken" / "processed"
        interim.mkdir(parents=True)
        processed.mkdir(parents=True)
        (interim / "lf.pkl").write_bytes(b"")
        (processed / "tsa.db").write_bytes(b"")

        with self.assertRaises(FileNotFoundError):
            _dataset_paths(self.data_dir, "broken", topology_variant="lines_only")


if __name__ == "__main__":
    unittest.main()
