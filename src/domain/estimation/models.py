from pydantic import BaseModel, Field


class Stats(BaseModel):
    """Quality/diagnostic indicators for data-science and debugging use, not calibrated
    for or intended as primary end-user output."""

    # Normalized pairwise compactness proxy, not a calibrated density estimate.
    neighborhood_compactness: float | None = Field(
        ...,
        description="Mean pairwise kernel similarity within the current group.",
    )
    n: int
    n_eff: float = Field(
        ...,
        description=(
            "Effective number of contributing simulation records within the current group, "
            "not the number of unique pre-fault states."
        ),
    )
    distances: dict[str, float]


class ReportStats(Stats):
    location_weight_mass: dict[str, float]
    location_counts: dict[str, int]


class LocationReportStats(Stats):
    weight_mass: float = Field(
        ...,
        description="Total generator-normalized neighbor weight assigned to this location.",
    )


class ReportSummary(BaseModel):
    cct_weighted: float
    cct_weighted_per_location: dict[str, float]
    stats: ReportStats


class ReportNeighbor(BaseModel):
    state: str
    cct: float
    location: str
    terminal: str | None
    type: str | None
    weight: float
    distance: float


class Report(BaseModel):
    summary: ReportSummary
    included_state_ids: list[str]
    per_neighbor: list[ReportNeighbor]


class LocationReportSummary(BaseModel):
    cct_weighted: float
    stats: LocationReportStats


class LocationReport(BaseModel):
    summary: LocationReportSummary
    included_state_ids: list[str]
    per_neighbor: list[ReportNeighbor]


class FsaReportNeighbor(BaseModel):
    state: str
    # Whichever FSA metric columns this dataset's fsa.pkl has (e.g. minF/maxF/maxRoCoF,
    # plus M1/M2/M3 for eles/2026-01 but not eles/2026-06) - kept generic rather than
    # named fields so datasets with different metric sets don't need different models.
    metrics: dict[str, float]
    weight: float
    distance: float


class FsaReportSummary(BaseModel):
    metrics_weighted: dict[str, float]
    stats: Stats


class FsaReport(BaseModel):
    summary: FsaReportSummary
    included_state_ids: list[str]
    per_neighbor: list[FsaReportNeighbor]
