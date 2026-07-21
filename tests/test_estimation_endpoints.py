import math
import unittest

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.estimate import router
from src.domain.estimation.models import (
    FsaReport,
    FsaReportNeighbor,
    FsaReportSummary,
    LocationReport,
    LocationReportStats,
    LocationReportSummary,
    Report,
    ReportNeighbor,
    ReportStats,
    ReportSummary,
    Stats,
)
from src.domain.estimation.service import EstimationService
from src.services.qdrant.repository import QueryResult


class _IdentityScaler:
    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.copy()


class _FakeRecordStore:
    """Stands in for SqliteRecordStore - filters a fixture DataFrame by state in-memory
    instead of hitting a real SQLite file, keeping unit tests fast."""

    def __init__(self, frame: pd.DataFrame):
        self._frame = frame
        self.columns = list(frame.columns)

    def fetch(self, state_ids) -> pd.DataFrame:
        ids = {str(s) for s in state_ids}
        return self._frame[self._frame["state"].astype(str).isin(ids)].reset_index(drop=True)


class _FakeDb:
    embed_cols = ["x"]

    def __init__(self):
        self.query_count = 0
        self.last_limit: int | None = None

    def query(self, *, state: pd.DataFrame, exclude_source_index, limit: int | None = None):
        self.query_count += 1
        self.last_limit = limit
        rows = pd.DataFrame(
            {"x": [0.0, 1.0, 3.0, 0.0]},
            index=["s1", "s2", "s3", "s4"],
        )
        rows.index.name = "state"
        return QueryResult(scores=np.array([], dtype=np.float64), rows=rows)


def _service() -> EstimationService:
    tsa = pd.DataFrame(
        {
            "state": ["s1", "s2", "s3", "s4"],
            "CCT": [1.0, 3.0, 100.0, 10.0],
            "Location": ["L1", "L1", "L2", "L1"],
            "Crit_gen": ["G1", "G1", "G1", "G2"],
            "Terminal": ["T1", "T1", "T2", "T3"],
            "Type": ["line", "line", "line", "line"],
        }
    )
    return EstimationService(columns=["x"], scaler=_IdentityScaler(), tsa=_FakeRecordStore(tsa), db=_FakeDb())


def _service_with_fsa() -> EstimationService:
    tsa = pd.DataFrame({"state": ["s1"], "CCT": [1.0], "Location": ["L1"], "Crit_gen": ["G1"], "Terminal": ["T1"]})
    # s4 is intentionally absent here (present in _FakeDb's retrieved rows) to verify a
    # retrieved state with no FSA coverage is silently excluded, not treated as an error.
    fsa = pd.DataFrame(
        {
            "state": ["s1", "s2", "s3"],
            "failed_gen": ["FG1", "FG1", "FG2"],
            "measured_gen": ["MG1", "MG1", "MG2"],
            "minF": [0.99, 0.98, 0.97],
            "maxF": [1.01, 1.02, 1.03],
        }
    )
    return EstimationService(
        columns=["x"],
        scaler=_IdentityScaler(),
        tsa=_FakeRecordStore(tsa),
        db=_FakeDb(),
        fsa=_FakeRecordStore(fsa),
    )


def _report() -> Report:
    return Report(
        summary=ReportSummary(
            cct_weighted=1.5,
            cct_weighted_per_location={"L1": 1.5},
            stats=ReportStats(
                location_weight_mass={"L1": 1.0},
                neighborhood_compactness=None,
                n=1,
                n_eff=1.0,
                distances={"min": 0.0, "mean": 0.0, "median": 0.0, "spread": 0.0, "norm": 0.0},
                location_counts={"L1": 1},
            ),
        ),
        included_state_ids=["s1"],
        per_neighbor=[
            ReportNeighbor(
                state="s1",
                cct=1.5,
                location="L1",
                terminal="T1",
                type="line",
                weight=1.0,
                distance=0.0,
            )
        ],
    )


def _location_report() -> LocationReport:
    return LocationReport(
        summary=LocationReportSummary(
            cct_weighted=1.5,
            stats=LocationReportStats(
                weight_mass=1.0,
                neighborhood_compactness=None,
                n=1,
                n_eff=1.0,
                distances={"min": 0.0, "mean": 0.0, "median": 0.0, "spread": 0.0, "norm": 0.0},
            ),
        ),
        included_state_ids=["s1"],
        per_neighbor=_report().per_neighbor,
    )


def _fsa_report() -> FsaReport:
    return FsaReport(
        summary=FsaReportSummary(
            metrics_weighted={"minF": 0.99, "maxF": 1.01},
            stats=Stats(
                neighborhood_compactness=None,
                n=1,
                n_eff=1.0,
                distances={"min": 0.0, "mean": 0.0, "median": 0.0, "spread": 0.0, "norm": 0.0},
            ),
        ),
        included_state_ids=["s1"],
        per_neighbor=[FsaReportNeighbor(state="s1", metrics={"minF": 0.99, "maxF": 1.01}, weight=1.0, distance=0.0)],
    )


class _RouteService:
    columns = ["x"]

    def ensure_columns(self, request_cols):
        if set(request_cols) != {"x"}:
            raise ValueError("bad columns")

    def estimate_by_generator(self, *, state, exclude_uids, n_neighbors=None):
        return {"G1": _report()}

    def estimate_by_location(self, *, state, exclude_uids, n_neighbors=None):
        return {"L1": {"G1": _location_report()}}

    def estimate_by_observed_generator(self, *, state, exclude_uids, n_neighbors=None):
        return {"MG1": {"FG1": _fsa_report()}}

    def estimate_by_failed_generator(self, *, state, exclude_uids, n_neighbors=None):
        return {"FG1": {"MG1": _fsa_report()}}


class _RouteServiceWithoutFsa(_RouteService):
    def estimate_by_observed_generator(self, *, state, exclude_uids, n_neighbors=None):
        raise NotImplementedError("This dataset does not provide FSA data")

    def estimate_by_failed_generator(self, *, state, exclude_uids, n_neighbors=None):
        raise NotImplementedError("This dataset does not provide FSA data")


class EstimationServiceEndpointTests(unittest.TestCase):
    def test_estimate_by_generator_matches_existing_generator_first_behavior(self):
        service = _service()

        by_generator = service.estimate_by_generator({"x": 0.0}, exclude_uids=[])
        legacy_alias = service.estimate({"x": 0.0}, exclude_uids=[])

        self.assertEqual(by_generator, legacy_alias)
        self.assertEqual(set(by_generator), {"G1", "G2"})
        self.assertEqual(by_generator["G1"].summary.stats.location_counts, {"L1": 2, "L2": 1})

    def test_n_neighbors_is_forwarded_to_db_query_and_defaults_to_none(self):
        db = _FakeDb()
        tsa = pd.DataFrame(
            {
                "state": ["s1", "s2", "s3", "s4"],
                "CCT": [1.0, 3.0, 100.0, 10.0],
                "Location": ["L1"] * 4,
                "Crit_gen": ["G1"] * 4,
            }
        )
        service = EstimationService(columns=["x"], scaler=_IdentityScaler(), tsa=_FakeRecordStore(tsa), db=db)

        service.estimate_by_generator({"x": 0.0}, exclude_uids=[])
        self.assertIsNone(db.last_limit)

        service.estimate_by_generator({"x": 0.0}, exclude_uids=[], n_neighbors=7)
        self.assertEqual(db.last_limit, 7)

    def test_estimate_by_location_nests_location_then_generator_and_uses_group_weights(self):
        service = _service()

        by_location = service.estimate_by_location({"x": 0.0}, exclude_uids=[])

        self.assertEqual(set(by_location), {"L1", "L2"})
        self.assertEqual(set(by_location["L1"]), {"G1", "G2"})

        l1_g1 = by_location["L1"]["G1"]
        expected_cct = (1.0 + math.exp(-1.0) * 3.0) / (1.0 + math.exp(-1.0))
        self.assertAlmostEqual(l1_g1.summary.cct_weighted, expected_cct)
        self.assertEqual(l1_g1.included_state_ids, ["s1", "s2"])
        self.assertEqual([neighbor.location for neighbor in l1_g1.per_neighbor], ["L1", "L1"])
        self.assertAlmostEqual(sum(neighbor.weight for neighbor in l1_g1.per_neighbor), 1.0)

        generator_weight_sum = 1.0 + math.exp(-1.0) + math.exp(-3.0)
        expected_location_mass = (1.0 + math.exp(-1.0)) / generator_weight_sum
        self.assertAlmostEqual(l1_g1.summary.stats.weight_mass, expected_location_mass)

    def test_estimate_fsa_raises_not_implemented_when_fsa_absent(self):
        service = _service()

        with self.assertRaises(NotImplementedError):
            service.estimate_by_observed_generator({"x": 0.0}, exclude_uids=[])
        with self.assertRaises(NotImplementedError):
            service.estimate_by_failed_generator({"x": 0.0}, exclude_uids=[])

    def test_estimate_by_failed_generator_groups_failed_then_measured_gen(self):
        service = _service_with_fsa()

        by_failed = service.estimate_by_failed_generator({"x": 0.0}, exclude_uids=[])

        self.assertEqual(set(by_failed), {"FG1", "FG2"})
        self.assertEqual(set(by_failed["FG1"]), {"MG1"})
        self.assertEqual(set(by_failed["FG2"]), {"MG2"})

        fg1 = by_failed["FG1"]["MG1"]
        expected_minf = (0.99 + math.exp(-1.0) * 0.98) / (1.0 + math.exp(-1.0))
        self.assertAlmostEqual(fg1.summary.metrics_weighted["minF"], expected_minf)
        self.assertEqual(set(fg1.summary.metrics_weighted), {"minF", "maxF"})
        self.assertEqual(fg1.included_state_ids, ["s1", "s2"])
        self.assertAlmostEqual(sum(neighbor.weight for neighbor in fg1.per_neighbor), 1.0)

        # s4 (retrieved by _FakeDb but absent from the fsa fixture) must not appear anywhere,
        # and must not have raised - a state lacking FSA coverage is silently excluded.
        all_included_states = {
            s for group in by_failed.values() for report in group.values() for s in report.included_state_ids
        }
        self.assertNotIn("s4", all_included_states)

    def test_estimate_by_observed_generator_groups_measured_then_failed_gen(self):
        service = _service_with_fsa()

        by_observed = service.estimate_by_observed_generator({"x": 0.0}, exclude_uids=[])

        self.assertEqual(set(by_observed), {"MG1", "MG2"})
        self.assertEqual(set(by_observed["MG1"]), {"FG1"})
        self.assertEqual(set(by_observed["MG2"]), {"FG2"})

        # Same underlying (failed_gen, measured_gen) pair as the failed-generator-primary
        # view above - just re-nested, so the numbers must match exactly.
        by_failed = service.estimate_by_failed_generator({"x": 0.0}, exclude_uids=[])
        self.assertEqual(
            by_observed["MG1"]["FG1"].summary.metrics_weighted,
            by_failed["FG1"]["MG1"].summary.metrics_weighted,
        )


class EstimationRouteTests(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(router)
        app.state.estimation_service = _RouteService()
        self.client = TestClient(app)

    def test_bare_estimate_endpoint_is_removed(self):
        response = self.client.post(
            "/api/v1/estimate",
            json={"variant": "1.0.0", "state": {"x": 0.0}, "exclude_uids": []},
        )

        self.assertEqual(response.status_code, 404)

    def test_pre_tsa_namespace_paths_are_removed(self):
        for path in ("/api/v1/estimate/by-generator", "/api/v1/estimate/by-location"):
            response = self.client.post(
                path,
                json={"variant": "1.0.0", "state": {"x": 0.0}, "exclude_uids": []},
            )
            self.assertEqual(response.status_code, 404, path)

    def test_by_generator_endpoint_returns_generator_first_response(self):
        response = self.client.post(
            "/api/v1/estimate/tsa/by-generator",
            json={"variant": "1.0.0", "state": {"x": 0.0}, "exclude_uids": []},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(response.json()["outputs"]), {"G1"})

    def test_by_location_endpoint_returns_location_first_response(self):
        response = self.client.post(
            "/api/v1/estimate/tsa/by-location",
            json={"variant": "1.0.0", "state": {"x": 0.0}, "exclude_uids": []},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(response.json()["outputs"]), {"L1"})
        self.assertEqual(set(response.json()["outputs"]["L1"]), {"G1"})

    def test_fsa_by_observed_generator_endpoint_returns_observed_first_response(self):
        response = self.client.post(
            "/api/v1/estimate/fsa/by-observed-generator",
            json={"variant": "1.0.0", "state": {"x": 0.0}, "exclude_uids": []},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(response.json()["outputs"]), {"MG1"})
        self.assertEqual(set(response.json()["outputs"]["MG1"]), {"FG1"})

    def test_fsa_by_failed_generator_endpoint_returns_failed_first_response(self):
        response = self.client.post(
            "/api/v1/estimate/fsa/by-failed-generator",
            json={"variant": "1.0.0", "state": {"x": 0.0}, "exclude_uids": []},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(response.json()["outputs"]), {"FG1"})
        self.assertEqual(set(response.json()["outputs"]["FG1"]), {"MG1"})


class EstimationFsaUnavailableRouteTests(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(router)
        app.state.estimation_service = _RouteServiceWithoutFsa()
        self.client = TestClient(app)

    def test_fsa_endpoints_return_501_when_dataset_has_no_fsa(self):
        for path in ("/api/v1/estimate/fsa/by-observed-generator", "/api/v1/estimate/fsa/by-failed-generator"):
            response = self.client.post(
                path,
                json={"variant": "1.0.0", "state": {"x": 0.0}, "exclude_uids": []},
            )
            self.assertEqual(response.status_code, 501, path)


if __name__ == "__main__":
    unittest.main()
