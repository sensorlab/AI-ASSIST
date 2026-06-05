from collections.abc import Mapping
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field
from pydantic_extra_types.semantic_version import SemanticVersion

from src.domain.estimation.models import LocationReport, Report
from src.domain.estimation.service import EstimationService


class StateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    variant: SemanticVersion = SemanticVersion.parse("1.0.0")
    state: Mapping[str, float | None]
    exclude_uids: frozenset[str] = Field(default_factory=frozenset)


class StateResponse(BaseModel):
    inputs: StateRequest
    outputs: dict[str, Report]


class LocationStateResponse(BaseModel):
    inputs: StateRequest
    outputs: dict[str, dict[str, LocationReport]]


router = APIRouter(prefix="/api/v1", tags=["estimate"])


@router.post("/estimate/by-generator", response_model=StateResponse)
async def estimate_by_generator(req: StateRequest, request: Request) -> StateResponse:
    service: EstimationService = request.app.state.estimation_service
    try:
        service.ensure_columns(req.state.keys())
        outputs = service.estimate_by_generator(state=req.state, exclude_uids=req.exclude_uids)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return StateResponse(inputs=req, outputs=outputs)


@router.post("/estimate/by-location", response_model=LocationStateResponse)
async def estimate_by_location(req: StateRequest, request: Request) -> LocationStateResponse:
    service: EstimationService = request.app.state.estimation_service
    try:
        service.ensure_columns(req.state.keys())
        outputs = service.estimate_by_location(state=req.state, exclude_uids=req.exclude_uids)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return LocationStateResponse(inputs=req, outputs=outputs)


@router.get("/columns")
async def columns(request: Request) -> dict[str, Any]:
    service: EstimationService = request.app.state.estimation_service
    return {
        "columns": service.columns,
        "sample_state": dict.fromkeys(service.columns, True),
    }
