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
    n_unique_states: int = Field(
        ...,
        description=(
            "Number of distinct pre-fault states contributing to this group, as opposed to "
            "n/n_eff which count simulation records. A group can have high n_eff (or high "
            "neighborhood_compactness, since records sharing a parent state contribute zero "
            "pairwise distance) while resting on very few distinct states - this diagnostic "
            "makes that narrow-support case visible instead of letting record-level "
            "aggregation read as broad support."
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


class SssaNeighbor(BaseModel):
    """A single retrieved (state, mode_id) row for one generator - raw, unweighted. mode_id
    is a per-state local identifier only (mode indices aren't comparable across operating
    states, per the data dictionary) - never compare/aggregate it across different states."""

    state: str
    mode_id: int
    real_part: float
    imag_part: float
    # Whichever participation metric columns this dataset's sssa.pkl has (e.g. ObsMag_speed/
    # ParAng_Psi2q for eles/2026-06, plain ConMag/ObsMag/... for eles/2026-01) - kept generic
    # rather than named fields, same reasoning as FsaReportNeighbor.metrics.
    metrics: dict[str, float]
    distance: float
