import unittest

from src.services.qdrant.config import QdrantConfig


class TopologyVariantConfigTests(unittest.TestCase):
    """QdrantConfig.topology_variant and its effect on collection_name - see
    datasets/eles/2026-06/README.md's "Topology Variants" section for why the variant is
    always folded into the collection name (avoids a stale Qdrant collection populated
    under a different variant's topology_id values silently being reused, since
    DatabaseQdrant.fit(force=False) only (re)populates a collection that doesn't exist yet)."""

    def test_defaults_to_lines_only(self):
        config = QdrantConfig(_env_file=None)

        self.assertEqual(config.topology_variant, "lines_only")

    def test_normalizes_case_and_whitespace(self):
        config = QdrantConfig(_env_file=None, TOPOLOGY_VARIANT="  Full  ")

        self.assertEqual(config.topology_variant, "full")

    def test_empty_value_falls_back_to_default(self):
        config = QdrantConfig(_env_file=None, TOPOLOGY_VARIANT="")

        self.assertEqual(config.topology_variant, "lines_only")

    def test_collection_name_includes_topology_variant(self):
        config = QdrantConfig(_env_file=None, DATASET_NAME="eles/2026-06", TOPOLOGY_VARIANT="slovenia_only")

        self.assertEqual(config.collection_name, f"states_eles-2026-06_slovenia_only_{config.scaler_version}")

    def test_different_variants_produce_different_collection_names(self):
        base = {"_env_file": None, "DATASET_NAME": "eles/2026-06"}
        lines_only = QdrantConfig(**base, TOPOLOGY_VARIANT="lines_only")
        full = QdrantConfig(**base, TOPOLOGY_VARIANT="full")

        self.assertNotEqual(lines_only.collection_name, full.collection_name)

    def test_explicit_collection_override_bypasses_variant_suffix(self):
        config = QdrantConfig(_env_file=None, QDRANT_COLLECTION="explicit_name", TOPOLOGY_VARIANT="full")

        self.assertEqual(config.collection_name, "explicit_name")


class ScalerVersionConfigTests(unittest.TestCase):
    """QdrantConfig.scaler_version and its effect on collection_name - same rationale as
    topology_variant above (see TopologyVariantConfigTests): a scaler formula change alters
    stored vector values/dimensionality, so it must be folded into the collection name to avoid
    DatabaseQdrant.fit(force=False) silently reusing a collection populated under an old scaler."""

    def test_defaults_to_v4(self):
        config = QdrantConfig(_env_file=None)

        self.assertEqual(config.scaler_version, "v4")

    def test_collection_name_includes_scaler_version(self):
        config = QdrantConfig(_env_file=None, DATASET_NAME="bus39", SCALER_VERSION="v4")

        self.assertEqual(config.collection_name, "states_bus39_lines_only_v4")

    def test_different_scaler_versions_produce_different_collection_names(self):
        base = {"_env_file": None, "DATASET_NAME": "bus39"}
        v3 = QdrantConfig(**base, SCALER_VERSION="v3")
        v4 = QdrantConfig(**base, SCALER_VERSION="v4")

        self.assertNotEqual(v3.collection_name, v4.collection_name)


if __name__ == "__main__":
    unittest.main()
