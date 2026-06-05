import math
import unittest

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.estimate import router
from src.domain.estimation.models import (
    LocationReport,
    LocationReportSummary,
    Report,
    ReportNeighbor,
    ReportSummary,
)
from src.domain.estimation.service import EstimationService
from src.services.qdrant.repository import QueryResult


class _IdentityScaler:
    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.copy()


class _FakeDb:
    embed_cols = ["x"]

    def __init__(self):
        self.query_count = 0

    def query(self, *, state: pd.DataFrame, exclude_source_index):
        self.query_count += 1
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
    return EstimationService(columns=["x"], scaler=_IdentityScaler(), tsa=tsa, db=_FakeDb())


def _report() -> Report:
    return Report(
        summary=ReportSummary(
            cct_weighted=1.5,
            cct_weighted_per_location={"L1": 1.5},
            location_weight_mass={"L1": 1.0},
            neighborhood_compactness=None,
            n=1,
            n_eff=1.0,
            distances={"min": 0.0, "mean": 0.0, "median": 0.0, "spread": 0.0, "norm": 0.0},
            location_counts={"L1": 1},
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
            weight_mass=1.0,
            neighborhood_compactness=None,
            n=1,
            n_eff=1.0,
            distances={"min": 0.0, "mean": 0.0, "median": 0.0, "spread": 0.0, "norm": 0.0},
        ),
        included_state_ids=["s1"],
        per_neighbor=_report().per_neighbor,
    )


class _RouteService:
    columns = ["x"]

    def ensure_columns(self, request_cols):
        if set(request_cols) != {"x"}:
            raise ValueError("bad columns")

    def estimate_by_generator(self, *, state, exclude_uids):
        return {"G1": _report()}

    def estimate_by_location(self, *, state, exclude_uids):
        return {"L1": {"G1": _location_report()}}


class EstimationServiceEndpointTests(unittest.TestCase):
    def test_estimate_by_generator_matches_existing_generator_first_behavior(self):
        service = _service()

        by_generator = service.estimate_by_generator({"x": 0.0}, exclude_uids=[])
        legacy_alias = service.estimate({"x": 0.0}, exclude_uids=[])

        self.assertEqual(by_generator, legacy_alias)
        self.assertEqual(set(by_generator), {"G1", "G2"})
        self.assertEqual(by_generator["G1"].summary.location_counts, {"L1": 2, "L2": 1})

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
        self.assertAlmostEqual(l1_g1.summary.weight_mass, expected_location_mass)


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

    def test_by_generator_endpoint_returns_generator_first_response(self):
        response = self.client.post(
            "/api/v1/estimate/by-generator",
            json={"variant": "1.0.0", "state": {"x": 0.0}, "exclude_uids": []},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(response.json()["outputs"]), {"G1"})

    def test_by_location_endpoint_returns_location_first_response(self):
        response = self.client.post(
            "/api/v1/estimate/by-location",
            json={"variant": "1.0.0", "state": {"x": 0.0}, "exclude_uids": []},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(response.json()["outputs"]), {"L1"})
        self.assertEqual(set(response.json()["outputs"]["L1"]), {"G1"})


if __name__ == "__main__":
    unittest.main()
