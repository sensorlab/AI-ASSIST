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
    LocationGroupReport,
    LocationReport,
    LocationReportStats,
    LocationReportSummary,
    Report,
    ReportNeighbor,
    SssaNeighbor,
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


def _service_with_sssa() -> EstimationService:
    tsa = pd.DataFrame({"state": ["s1"], "CCT": [1.0], "Location": ["L1"], "Crit_gen": ["G1"], "Terminal": ["T1"]})
    # s1 has two modes (1 and 2), both observed by GenA - mode_id must not be used as a
    # group key, so both land under GenA alongside s2's mode. GenB only appears once, for
    # s1's mode 1. s4 (present in _FakeDb's retrieved rows) is intentionally absent here to
    # verify a retrieved state with no SSSA coverage is silently excluded, not an error.
    sssa = pd.DataFrame(
        {
            "state": ["s1", "s1", "s1", "s2"],
            "mode_id": [1, 1, 2, 1],
            "generator": ["GenA", "GenB", "GenA", "GenA"],
            "real_part": [-0.1, -0.1, -0.2, -0.3],
            "imag_part": [5.0, 5.0, 6.0, 7.0],
            "ObsMag": [0.10, 0.20, 0.30, 0.40],
        }
    )
    return EstimationService(
        columns=["x"],
        scaler=_IdentityScaler(),
        tsa=_FakeRecordStore(tsa),
        db=_FakeDb(),
        sssa=_FakeRecordStore(sssa),
    )


def _service_with_sssa_mode_shapes() -> EstimationService:
    """Dedicated fixture for matched_mode tests - unlike _service_with_sssa() above, this
    includes ParMag (participation magnitude) so cosine-similarity matching has real signal
    to work with. s1/mode1 and s2/mode1 share the same shape (dominant GenA, minor GenB) and
    a close eigenvalue - the intended cross-state match. s1/mode2 is deliberately shaped and
    positioned differently (GenA only, far-off eigenvalue) so it has no good cross-state
    candidate, exercising the "no confidence threshold" behavior."""
    tsa = pd.DataFrame({"state": ["s1"], "CCT": [1.0], "Location": ["L1"], "Crit_gen": ["G1"], "Terminal": ["T1"]})
    sssa = pd.DataFrame(
        {
            "state": ["s1", "s1", "s1", "s2", "s2"],
            "mode_id": [1, 1, 2, 1, 1],
            "generator": ["GenA", "GenB", "GenA", "GenA", "GenB"],
            "real_part": [-0.1, -0.1, -9.0, -0.15, -0.15],
            "imag_part": [5.0, 5.0, 50.0, 5.2, 5.2],
            "ObsMag": [0.10, 0.20, 0.30, 0.40, 0.45],
            "ParMag": [0.9, 0.1, 0.9, 0.85, 0.15],
        }
    )
    return EstimationService(
        columns=["x"],
        scaler=_IdentityScaler(),
        tsa=_FakeRecordStore(tsa),
        db=_FakeDb(),
        sssa=_FakeRecordStore(sssa),
    )


def _location_report() -> LocationReport:
    return LocationReport(
        summary=LocationReportSummary(
            cct_weighted=1.5,
            stats=LocationReportStats(
                weight_mass=1.0,
                weight_mass_mean=1.0,
                cct_weighted_std=None,
                cct_distance_correlation=None,
                cct_quantiles=None,
                neighborhood_compactness=None,
                n=1,
                n_eff=1.0,
                n_unique_states=1,
                distances={"min": 0.0, "mean": 0.0, "median": 0.0, "spread": 0.0, "norm": 0.0},
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


def _report() -> Report:
    return Report(
        location_likelihood={"L1": 1.0},
        per_location={"L1": _location_report()},
        included_state_ids=["s1"],
        neighbors=_location_report().per_neighbor,
    )


def _location_group_report() -> LocationGroupReport:
    return LocationGroupReport(
        crit_gen_likelihood={"G1": 1.0},
        per_crit_gen={"G1": _location_report()},
        included_state_ids=["s1"],
        neighbors=_location_report().per_neighbor,
    )


def _fsa_report() -> FsaReport:
    return FsaReport(
        summary=FsaReportSummary(
            metrics_weighted={"minF": 0.99, "maxF": 1.01},
            stats=Stats(
                neighborhood_compactness=None,
                n=1,
                n_eff=1.0,
                n_unique_states=1,
                distances={"min": 0.0, "mean": 0.0, "median": 0.0, "spread": 0.0, "norm": 0.0},
            ),
        ),
        included_state_ids=["s1"],
        per_neighbor=[FsaReportNeighbor(state="s1", metrics={"minF": 0.99, "maxF": 1.01}, weight=1.0, distance=0.0)],
    )


class _RouteService:
    columns = ["x"]
    default_n_neighbors = 100

    def ensure_columns(self, request_cols):
        if set(request_cols) != {"x"}:
            raise ValueError("bad columns")

    def estimate_by_generator(self, *, state, exclude_uids, n_neighbors=None):
        return {"G1": _report()}

    def estimate_by_location(self, *, state, exclude_uids, n_neighbors=None):
        return {"L1": _location_group_report()}

    def estimate_by_observed_generator(self, *, state, exclude_uids, n_neighbors=None):
        return {"MG1": {"FG1": _fsa_report()}}

    def estimate_by_failed_generator(self, *, state, exclude_uids, n_neighbors=None):
        return {"FG1": {"MG1": _fsa_report()}}

    def estimate_sssa_by_generator(self, *, state, exclude_uids, n_neighbors=None):
        return {
            "GenA": [
                SssaNeighbor(
                    state="s1", mode_id=1, real_part=-0.1, imag_part=5.0, metrics={}, matched_mode=None, distance=0.0
                )
            ]
        }


class _RouteServiceWithoutFsa(_RouteService):
    def estimate_by_observed_generator(self, *, state, exclude_uids, n_neighbors=None):
        raise NotImplementedError("This dataset does not provide FSA data")

    def estimate_by_failed_generator(self, *, state, exclude_uids, n_neighbors=None):
        raise NotImplementedError("This dataset does not provide FSA data")


class _RouteServiceWithoutSssa(_RouteService):
    def estimate_sssa_by_generator(self, *, state, exclude_uids, n_neighbors=None):
        raise NotImplementedError("This dataset does not provide SSSA (small-signal stability) data")


class EstimationServiceEndpointTests(unittest.TestCase):
    def test_estimate_by_generator_matches_existing_generator_first_behavior(self):
        service = _service()

        by_generator = service.estimate_by_generator({"x": 0.0}, exclude_uids=[])
        legacy_alias = service.estimate({"x": 0.0}, exclude_uids=[])

        self.assertEqual(by_generator, legacy_alias)
        self.assertEqual(set(by_generator), {"G1", "G2"})
        self.assertEqual(
            {loc: lr.summary.stats.n for loc, lr in by_generator["G1"].per_location.items()},
            {"L1": 2, "L2": 1},
        )

        # location_likelihood is sorted descending by weight_mass (a probability-like score
        # summing to 1.0 across the group) - L1 (2 close neighbors) must outrank L2 (1
        # farther neighbor).
        location_likelihood = by_generator["G1"].location_likelihood
        self.assertEqual(list(location_likelihood), ["L1", "L2"])
        self.assertAlmostEqual(sum(location_likelihood.values()), 1.0)

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
        self.assertEqual(set(by_location["L1"].per_crit_gen), {"G1", "G2"})

        l1_g1 = by_location["L1"].per_crit_gen["G1"]
        expected_cct = (1.0 + math.exp(-1.0) * 3.0) / (1.0 + math.exp(-1.0))
        self.assertAlmostEqual(l1_g1.summary.cct_weighted, expected_cct)
        self.assertEqual(l1_g1.included_state_ids, ["s1", "s2"])
        self.assertEqual([neighbor.location for neighbor in l1_g1.per_neighbor], ["L1", "L1"])
        self.assertAlmostEqual(sum(neighbor.weight for neighbor in l1_g1.per_neighbor), 1.0)

        generator_weight_sum = 1.0 + math.exp(-1.0) + math.exp(-3.0)
        expected_location_mass = (1.0 + math.exp(-1.0)) / generator_weight_sum
        self.assertAlmostEqual(l1_g1.summary.stats.weight_mass, expected_location_mass)
        self.assertAlmostEqual(l1_g1.summary.stats.weight_mass_mean, expected_location_mass / 2)

        # cct_weighted_std: weighted std of {CCT=1.0, CCT=3.0} under the same qw_norm used
        # for expected_cct above - the two neighbors disagree substantially on CCT (1.0 vs
        # 3.0s), so this should be a large, non-trivial spread, not near zero.
        qw = [1.0 / (1.0 + math.exp(-1.0)), math.exp(-1.0) / (1.0 + math.exp(-1.0))]
        expected_std = math.sqrt(sum(w * (cct - expected_cct) ** 2 for w, cct in zip(qw, [1.0, 3.0], strict=True)))
        self.assertAlmostEqual(l1_g1.summary.stats.cct_weighted_std, expected_std)

        # L2/G1 has a single neighbor (s3) - std is undefined, not misleadingly 0.0.
        l2_g1 = by_location["L2"].per_crit_gen["G1"]
        self.assertIsNone(l2_g1.summary.stats.cct_weighted_std)
        self.assertIsNone(l2_g1.summary.stats.cct_distance_correlation)
        self.assertIsNone(l2_g1.summary.stats.cct_quantiles)

        # cct_distance_correlation: with exactly 2 points, a weighted correlation is always
        # a perfect +-1 (two points always determine a line exactly) - here CCT increases
        # with distance (s1 d=0/CCT=1.0, s2 d=1/CCT=3.0), so it must be +1, confirming the
        # sign/formula rather than just "some strong correlation."
        self.assertAlmostEqual(l1_g1.summary.stats.cct_distance_correlation, 1.0)

        # cct_quantiles: q10 lands on the closer/heavier-weighted neighbor's CCT (1.0), q90
        # on the farther/lighter one's (3.0) - the two extremes of what's actually present.
        self.assertAlmostEqual(l1_g1.summary.stats.cct_quantiles["q10"], 1.0)
        self.assertAlmostEqual(l1_g1.summary.stats.cct_quantiles["q90"], 3.0)

        # crit_gen_likelihood: raw (never renormalized) kernel mass per generator at L1 -
        # G1's two neighbors (s1 d=0, s2 d=1) vs G2's one neighbor (s4 d=0) - G1's mass
        # (1 + exp(-1)) exceeds G2's (1), so G1 must rank first despite weight_mass (a
        # different, non-comparable-across-generators quantity) not being usable here.
        raw_g1 = 1.0 + math.exp(-1.0)
        raw_g2 = 1.0
        total_raw = raw_g1 + raw_g2
        location_likelihood = by_location["L1"].crit_gen_likelihood
        self.assertEqual(list(location_likelihood), ["G1", "G2"])
        self.assertAlmostEqual(location_likelihood["G1"], raw_g1 / total_raw)
        self.assertAlmostEqual(location_likelihood["G2"], raw_g2 / total_raw)
        self.assertAlmostEqual(sum(location_likelihood.values()), 1.0)

        # L2 has only one generator (G1) - trivially the sole, most-likely entry.
        self.assertEqual(by_location["L2"].crit_gen_likelihood, {"G1": 1.0})

        # Top-level neighbors/included_state_ids are location-wide (across all generators).
        self.assertEqual(by_location["L1"].included_state_ids, ["s1", "s2", "s4"])

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

    def test_estimate_sssa_raises_not_implemented_when_sssa_absent(self):
        service = _service()

        with self.assertRaises(NotImplementedError):
            service.estimate_sssa_by_generator({"x": 0.0}, exclude_uids=[])

    def test_estimate_sssa_by_generator_groups_by_generator_not_mode_and_is_unweighted(self):
        service = _service_with_sssa()

        by_generator = service.estimate_sssa_by_generator({"x": 0.0}, exclude_uids=[])

        # mode_id is never a group key - GenA pools both of s1's modes (1 and 2) plus s2's
        # mode, all under the same generator; GenB only has one entry.
        self.assertEqual(set(by_generator), {"GenA", "GenB"})
        self.assertEqual(len(by_generator["GenA"]), 3)
        self.assertEqual(len(by_generator["GenB"]), 1)

        # Raw, unweighted values - every retrieved row appears untouched, not averaged.
        gen_a_modes = {n.mode_id for n in by_generator["GenA"]}
        self.assertEqual(gen_a_modes, {1, 2})
        gen_a_states = {n.state for n in by_generator["GenA"]}
        self.assertEqual(gen_a_states, {"s1", "s2"})

        # Sorted by distance ascending - s1 (x=0.0) is closer to the query (x=0.0) than s2
        # (x=1.0), so both s1 rows must come before the s2 row.
        self.assertEqual([n.state for n in by_generator["GenA"]], ["s1", "s1", "s2"])
        self.assertAlmostEqual(by_generator["GenA"][0].distance, 0.0)
        self.assertAlmostEqual(by_generator["GenA"][2].distance, 1.0)

        # real_part/imag_part repeat exactly as recorded per (state, mode_id) - no weighting.
        mode1 = next(n for n in by_generator["GenA"] if n.mode_id == 1 and n.state == "s1")
        self.assertAlmostEqual(mode1.real_part, -0.1)
        self.assertAlmostEqual(mode1.imag_part, 5.0)
        self.assertAlmostEqual(mode1.metrics["ObsMag"], 0.10)

        gen_b = by_generator["GenB"][0]
        self.assertAlmostEqual(gen_b.metrics["ObsMag"], 0.20)

        # s4 (retrieved by _FakeDb but absent from the sssa fixture) must not appear anywhere,
        # and must not have raised - a state lacking SSSA coverage is silently excluded.
        all_states = {n.state for neighbors in by_generator.values() for n in neighbors}
        self.assertNotIn("s4", all_states)

    def test_estimate_sssa_by_generator_attaches_best_cross_state_mode_match(self):
        service = _service_with_sssa_mode_shapes()

        by_generator = service.estimate_sssa_by_generator({"x": 0.0}, exclude_uids=[])

        # s1/mode1 (dominant GenA, minor GenB) is shaped like s2/mode1 and close in
        # eigenvalue too - both cosine and eigenvalue agree it's the best cross-state match.
        mode_s1_1 = next(n for n in by_generator["GenA"] if n.state == "s1" and n.mode_id == 1)
        self.assertIsNotNone(mode_s1_1.matched_mode)
        self.assertEqual(mode_s1_1.matched_mode.state, "s2")
        self.assertEqual(mode_s1_1.matched_mode.mode_id, 1)

        # The match is symmetric here since s2/mode1 only has one cross-state candidate.
        mode_s2_1 = next(n for n in by_generator["GenA"] if n.state == "s2" and n.mode_id == 1)
        self.assertIsNotNone(mode_s2_1.matched_mode)
        self.assertEqual(mode_s2_1.matched_mode.state, "s1")
        self.assertEqual(mode_s2_1.matched_mode.mode_id, 1)

        # matched_mode is computed per (state, mode_id) and repeats across every generator
        # row sharing that mode - GenB's s1/mode1 row must report the same match as GenA's.
        mode_s1_1_genb = by_generator["GenB"][0]
        self.assertEqual(mode_s1_1_genb.state, "s1")
        self.assertEqual(mode_s1_1_genb.mode_id, 1)
        self.assertEqual(mode_s1_1_genb.matched_mode.state, "s2")
        self.assertEqual(mode_s1_1_genb.matched_mode.mode_id, 1)

        # s1/mode2 is shaped completely differently (GenA only, eigenvalue far away) - its
        # only cross-state candidate is still s2/mode1 (the only other state retrieved), and
        # it must still be returned despite being a poor match: no confidence threshold is
        # applied, so a bad match is surfaced with a large eigenvalue_distance rather than
        # silently dropped.
        mode_s1_2 = next(n for n in by_generator["GenA"] if n.state == "s1" and n.mode_id == 2)
        self.assertIsNotNone(mode_s1_2.matched_mode)
        self.assertEqual(mode_s1_2.matched_mode.state, "s2")
        self.assertGreater(mode_s1_2.matched_mode.eigenvalue_distance, mode_s1_1.matched_mode.eigenvalue_distance)

    def test_sssa_mode_match_is_none_without_a_second_state(self):
        tsa = pd.DataFrame({"state": ["s1"], "CCT": [1.0], "Location": ["L1"], "Crit_gen": ["G1"], "Terminal": ["T1"]})
        sssa = pd.DataFrame(
            {
                "state": ["s1", "s1"],
                "mode_id": [1, 2],
                "generator": ["GenA", "GenA"],
                "real_part": [-0.1, -0.2],
                "imag_part": [5.0, 6.0],
                "ObsMag": [0.1, 0.2],
                "ParMag": [0.9, 0.8],
            }
        )
        service = EstimationService(
            columns=["x"],
            scaler=_IdentityScaler(),
            tsa=_FakeRecordStore(tsa),
            db=_FakeDb(),
            sssa=_FakeRecordStore(sssa),
        )

        by_generator = service.estimate_sssa_by_generator({"x": 0.0}, exclude_uids=[])

        # Only s1 has SSSA coverage (s2/s3/s4 are absent from the fixture) - every mode's
        # only candidates are within its own state, so nothing can match.
        self.assertTrue(all(n.matched_mode is None for n in by_generator["GenA"]))

    def test_sssa_mode_match_handles_single_retrieved_mode_without_error(self):
        tsa = pd.DataFrame({"state": ["s1"], "CCT": [1.0], "Location": ["L1"], "Crit_gen": ["G1"], "Terminal": ["T1"]})
        sssa = pd.DataFrame(
            {
                "state": ["s1"],
                "mode_id": [1],
                "generator": ["GenA"],
                "real_part": [-0.1],
                "imag_part": [5.0],
                "ObsMag": [0.1],
                "ParMag": [0.9],
            }
        )
        service = EstimationService(
            columns=["x"],
            scaler=_IdentityScaler(),
            tsa=_FakeRecordStore(tsa),
            db=_FakeDb(),
            sssa=_FakeRecordStore(sssa),
        )

        by_generator = service.estimate_sssa_by_generator({"x": 0.0}, exclude_uids=[])

        self.assertIsNone(by_generator["GenA"][0].matched_mode)


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

    def test_max_states_echoes_resolved_default_when_omitted(self):
        response = self.client.post(
            "/api/v1/estimate/tsa/by-generator",
            json={"variant": "1.0.0", "state": {"x": 0.0}, "exclude_uids": []},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["inputs"]["max_states"], _RouteService.default_n_neighbors)

    def test_max_states_echoes_caller_value_when_provided(self):
        response = self.client.post(
            "/api/v1/estimate/tsa/by-generator",
            json={"variant": "1.0.0", "state": {"x": 0.0}, "exclude_uids": [], "max_states": 7},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["inputs"]["max_states"], 7)

    def test_max_states_above_upper_bound_is_rejected(self):
        response = self.client.post(
            "/api/v1/estimate/tsa/by-generator",
            json={"variant": "1.0.0", "state": {"x": 0.0}, "exclude_uids": [], "max_states": 501},
        )

        # Pydantic validation, not the route body - a 422, not the 400 estimate_by_generator
        # itself raises for bad state columns.
        self.assertEqual(response.status_code, 422)

    def test_by_location_endpoint_returns_location_first_response(self):
        response = self.client.post(
            "/api/v1/estimate/tsa/by-location",
            json={"variant": "1.0.0", "state": {"x": 0.0}, "exclude_uids": []},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(response.json()["outputs"]), {"L1"})
        self.assertEqual(set(response.json()["outputs"]["L1"]["per_crit_gen"]), {"G1"})

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

    def test_sssa_by_generator_endpoint_returns_generator_first_response(self):
        response = self.client.post(
            "/api/v1/estimate/sssa/by-generator",
            json={"variant": "1.0.0", "state": {"x": 0.0}, "exclude_uids": []},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(response.json()["outputs"]), {"GenA"})
        self.assertEqual(response.json()["outputs"]["GenA"][0]["mode_id"], 1)


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


class EstimationSssaUnavailableRouteTests(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(router)
        app.state.estimation_service = _RouteServiceWithoutSssa()
        self.client = TestClient(app)

    def test_sssa_endpoint_returns_501_when_dataset_has_no_sssa(self):
        response = self.client.post(
            "/api/v1/estimate/sssa/by-generator",
            json={"variant": "1.0.0", "state": {"x": 0.0}, "exclude_uids": []},
        )

        self.assertEqual(response.status_code, 501)


if __name__ == "__main__":
    unittest.main()
