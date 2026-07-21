from collections.abc import Mapping
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field
from pydantic_extra_types.semantic_version import SemanticVersion

from src.domain.estimation.models import FsaReport, LocationReport, Report
from src.domain.estimation.service import EstimationService


class StateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    variant: SemanticVersion = SemanticVersion.parse("1.0.0")
    state: Mapping[str, float | None]
    exclude_uids: frozenset[str] = Field(default_factory=frozenset)
    n_neighbors: int | None = Field(
        default=None,
        gt=0,
        description="Number of nearest neighbors to retrieve from Qdrant. Defaults to the service's built-in limit.",
    )


class StateResponse(BaseModel):
    inputs: StateRequest
    outputs: dict[str, Report]


class LocationStateResponse(BaseModel):
    inputs: StateRequest
    outputs: dict[str, dict[str, LocationReport]]


class FsaStateResponse(BaseModel):
    inputs: StateRequest
    outputs: dict[str, dict[str, FsaReport]]


router = APIRouter(prefix="/api/v1", tags=["estimate"])


@router.post("/estimate/tsa/by-generator", response_model=StateResponse)
async def estimate_by_generator(req: StateRequest, request: Request) -> StateResponse:
    service: EstimationService = request.app.state.estimation_service
    try:
        service.ensure_columns(req.state.keys())
        outputs = service.estimate_by_generator(
            state=req.state, exclude_uids=req.exclude_uids, n_neighbors=req.n_neighbors
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return StateResponse(inputs=req, outputs=outputs)


@router.post("/estimate/tsa/by-location", response_model=LocationStateResponse)
async def estimate_by_location(req: StateRequest, request: Request) -> LocationStateResponse:
    service: EstimationService = request.app.state.estimation_service
    try:
        service.ensure_columns(req.state.keys())
        outputs = service.estimate_by_location(
            state=req.state, exclude_uids=req.exclude_uids, n_neighbors=req.n_neighbors
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return LocationStateResponse(inputs=req, outputs=outputs)


@router.post("/estimate/fsa/by-observed-generator", response_model=FsaStateResponse)
async def estimate_fsa_by_observed_generator(req: StateRequest, request: Request) -> FsaStateResponse:
    service: EstimationService = request.app.state.estimation_service
    try:
        service.ensure_columns(req.state.keys())
        outputs = service.estimate_by_observed_generator(
            state=req.state, exclude_uids=req.exclude_uids, n_neighbors=req.n_neighbors
        )
    except NotImplementedError as exc:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return FsaStateResponse(inputs=req, outputs=outputs)


@router.post("/estimate/fsa/by-failed-generator", response_model=FsaStateResponse)
async def estimate_fsa_by_failed_generator(req: StateRequest, request: Request) -> FsaStateResponse:
    service: EstimationService = request.app.state.estimation_service
    try:
        service.ensure_columns(req.state.keys())
        outputs = service.estimate_by_failed_generator(
            state=req.state, exclude_uids=req.exclude_uids, n_neighbors=req.n_neighbors
        )
    except NotImplementedError as exc:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return FsaStateResponse(inputs=req, outputs=outputs)


@router.get("/columns")
async def columns(request: Request) -> dict[str, Any]:
    service: EstimationService = request.app.state.estimation_service
    return {
        "columns": service.columns,
        "sample_state": dict.fromkeys(service.columns, True),
    }
