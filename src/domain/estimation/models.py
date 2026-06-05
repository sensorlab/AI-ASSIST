from pydantic import BaseModel, Field


class ReportSummary(BaseModel):
    cct_weighted: float
    cct_weighted_per_location: dict[str, float]
    location_weight_mass: dict[str, float]
    # Normalized pairwise compactness proxy, not a calibrated density estimate.
    neighborhood_compactness: float | None = Field(
        ...,
        description="Mean pairwise kernel similarity within the current critical-generator group.",
    )
    n: int
    n_eff: float = Field(
        ...,
        description=(
            "Effective number of contributing simulation records within the current critical-generator group, "
            "not the number of unique pre-fault states."
        ),
    )
    distances: dict[str, float]
    location_counts: dict[str, int]


class ReportNeighbor(BaseModel):
    state: str
    cct: float
    location: str
    terminal: str
    type: str
    weight: float
    distance: float


class Report(BaseModel):
    summary: ReportSummary
    included_state_ids: list[str]
    per_neighbor: list[ReportNeighbor]


class LocationReportSummary(BaseModel):
    cct_weighted: float
    weight_mass: float = Field(
        ...,
        description="Total generator-normalized neighbor weight assigned to this location.",
    )
    # Normalized pairwise compactness proxy, not a calibrated density estimate.
    neighborhood_compactness: float | None = Field(
        ...,
        description="Mean pairwise kernel similarity within the current location/generator group.",
    )
    n: int
    n_eff: float = Field(
        ...,
        description=(
            "Effective number of contributing simulation records within the current location/generator group, "
            "not the number of unique pre-fault states."
        ),
    )
    distances: dict[str, float]


class LocationReport(BaseModel):
    summary: LocationReportSummary
    included_state_ids: list[str]
    per_neighbor: list[ReportNeighbor]
