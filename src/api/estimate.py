from collections.abc import Mapping
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field
from pydantic_extra_types.semantic_version import SemanticVersion

from src.domain.estimation.models import FsaReport, LocationGroupReport, Report, SssaNeighbor
from src.domain.estimation.service import EstimationService


class StateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    variant: SemanticVersion = SemanticVersion.parse("1.0.0")
    state: Mapping[str, float | None]
    exclude_uids: frozenset[str] = Field(default_factory=frozenset)
    max_states: int | None = Field(
        default=None,
        gt=0,
        le=500,
        description=(
            "Cap on the number of nearest states to retrieve from Qdrant (fewer may come "
            "back if the topology-filtered candidate pool is smaller than this). Defaults "
            "to the service's built-in limit when omitted - the response's inputs.max_states "
            "always shows the actual value used, even when this was left unset. Upper-bounded "
            "at 500: SSSA's cross-state mode matching (estimate/sssa/by-generator) is roughly "
            "quadratic in the number of retrieved SSSA rows, and eles/2026-01's ~179 modes/state "
            "means an unbounded max_states could tie up a worker for a very long time - 500 "
            "keeps the worst case on the same order as the already-measured eles/2026-01 "
            "full-corpus benchmark (~90k modes, ~100s), not open-ended."
        ),
    )


class StateResponse(BaseModel):
    inputs: StateRequest
    outputs: dict[str, Report]


class LocationStateResponse(BaseModel):
    inputs: StateRequest
    outputs: dict[str, LocationGroupReport]


class FsaStateResponse(BaseModel):
    inputs: StateRequest
    outputs: dict[str, dict[str, FsaReport]]


class SssaStateResponse(BaseModel):
    inputs: StateRequest
    outputs: dict[str, list[SssaNeighbor]]


router = APIRouter(prefix="/api/v1", tags=["estimate"])


def _resolve_max_states(req: StateRequest, service: EstimationService) -> StateRequest:
    """Returns req as-is if max_states was set explicitly, otherwise a copy with it filled
    in from the service's actual default - so the echoed inputs in the response always show
    what was actually queried, useful for debugging or later analysis, rather than a bare
    None that doesn't say what limit was actually applied."""
    if req.max_states is not None:
        return req
    return req.model_copy(update={"max_states": service.default_n_neighbors})


@router.post("/estimate/tsa/by-generator", response_model=StateResponse)
async def estimate_by_generator(req: StateRequest, request: Request) -> StateResponse:
    """For each critical generator retrieved near the query state: location_likelihood
    ranks the fault locations found by how strongly they're supported, and per_location
    gives the full CCT estimate + supporting detail behind each one."""
    service: EstimationService = request.app.state.estimation_service
    try:
        service.ensure_columns(req.state.keys())
        outputs = service.estimate_by_generator(
            state=req.state, exclude_uids=req.exclude_uids, n_neighbors=req.max_states
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return StateResponse(inputs=_resolve_max_states(req, service), outputs=outputs)


@router.post("/estimate/tsa/by-location", response_model=LocationStateResponse)
async def estimate_by_location(req: StateRequest, request: Request) -> LocationStateResponse:
    """The mirror image of by-generator: for each fault location retrieved near the query
    state, crit_gen_likelihood ranks which critical generator is most likely there, and
    per_crit_gen gives the same full per-generator detail."""
    service: EstimationService = request.app.state.estimation_service
    try:
        service.ensure_columns(req.state.keys())
        outputs = service.estimate_by_location(
            state=req.state, exclude_uids=req.exclude_uids, n_neighbors=req.max_states
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return LocationStateResponse(inputs=_resolve_max_states(req, service), outputs=outputs)


@router.post("/estimate/fsa/by-observed-generator", response_model=FsaStateResponse)
async def estimate_fsa_by_observed_generator(req: StateRequest, request: Request) -> FsaStateResponse:
    """Frequency stability (FSA), primary view: for each observed/measured generator, the
    outcome per failed generator. Returns HTTP 501 if the active dataset has no FSA data."""
    service: EstimationService = request.app.state.estimation_service
    try:
        service.ensure_columns(req.state.keys())
        outputs = service.estimate_by_observed_generator(
            state=req.state, exclude_uids=req.exclude_uids, n_neighbors=req.max_states
        )
    except NotImplementedError as exc:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return FsaStateResponse(inputs=_resolve_max_states(req, service), outputs=outputs)


@router.post("/estimate/fsa/by-failed-generator", response_model=FsaStateResponse)
async def estimate_fsa_by_failed_generator(req: StateRequest, request: Request) -> FsaStateResponse:
    """Frequency stability (FSA), secondary view: for each failed generator, the outcome
    per observed/measured generator. Returns HTTP 501 if the active dataset has no FSA data."""
    service: EstimationService = request.app.state.estimation_service
    try:
        service.ensure_columns(req.state.keys())
        outputs = service.estimate_by_failed_generator(
            state=req.state, exclude_uids=req.exclude_uids, n_neighbors=req.max_states
        )
    except NotImplementedError as exc:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return FsaStateResponse(inputs=_resolve_max_states(req, service), outputs=outputs)


@router.post("/estimate/sssa/by-generator", response_model=SssaStateResponse)
async def estimate_sssa_by_generator(req: StateRequest, request: Request) -> SssaStateResponse:
    """Small-signal stability (SSSA), grouped by generator (the only identity comparable
    across states - mode_id is per-state local and never used as a grouping key). Raw and
    unweighted by design: every retrieved (state, mode_id) row is returned as-is, sorted by
    distance, pending domain input on what aggregation (if any) is wanted. Returns HTTP 501 if
    the active dataset has no SSSA data."""
    service: EstimationService = request.app.state.estimation_service
    try:
        service.ensure_columns(req.state.keys())
        outputs = service.estimate_sssa_by_generator(
            state=req.state, exclude_uids=req.exclude_uids, n_neighbors=req.max_states
        )
    except NotImplementedError as exc:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return SssaStateResponse(inputs=_resolve_max_states(req, service), outputs=outputs)


@router.get("/columns")
async def columns(request: Request) -> dict[str, Any]:
    """The full list of grid-state feature columns this dataset expects in `state`, plus a
    sample_state stub (all values True) showing the expected shape."""
    service: EstimationService = request.app.state.estimation_service
    return {
        "columns": service.columns,
        "sample_state": dict.fromkeys(service.columns, True),
    }
