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


class LocationReportStats(Stats):
    weight_mass: float = Field(
        ...,
        description="Total generator-normalized neighbor weight assigned to this location.",
    )
    weight_mass_mean: float = Field(
        ...,
        description=(
            "weight_mass / n - the mean per-neighbor weight at this location, not the "
            "same thing as neighborhood_compactness (that's a pairwise measure among the "
            "neighbors themselves; this is the mean of query-to-neighbor weights)."
        ),
    )
    cct_weighted_std: float | None = Field(
        ...,
        description=(
            "Weighted standard deviation of CCT among this location's neighbors - do they "
            "agree on the outcome, not just look similar as inputs (that's what "
            "neighborhood_compactness measures instead). None below n=2, same convention as "
            "neighborhood_compactness, since a single neighbor's spread (0.0) would "
            "misleadingly read as high confidence rather than no data to compare against."
        ),
    )
    cct_distance_correlation: float | None = Field(
        ...,
        description=(
            "Weighted correlation between query-to-neighbor distance and CCT - does the "
            "outcome actually vary smoothly with distance in this neighborhood, or is the "
            "variation more like scatter? A strong correlation (either sign) means the "
            "distance-based weighting behind cct_weighted is doing real work (favoring the "
            "right neighbors); near-zero means CCT doesn't track distance here, so "
            "cct_weighted isn't buying much over a plain average. None below n=2 or when "
            "distance/CCT has zero variance within the group (correlation undefined)."
        ),
    )
    cct_quantiles: dict[str, float] | None = Field(
        ...,
        description=(
            "Weighted 10th/90th percentile of CCT among this location's neighbors - a "
            "shape-aware complement to cct_weighted_std, which assumes symmetric spread; "
            "an outlier neighbor could instead skew the distribution one way. None below "
            "n=2, same convention as cct_weighted_std."
        ),
    )


class ReportNeighbor(BaseModel):
    state: str
    cct: float
    location: str
    terminal: str | None
    type: str | None
    weight: float
    distance: float


class LocationReportSummary(BaseModel):
    cct_weighted: float
    stats: LocationReportStats


class LocationReport(BaseModel):
    summary: LocationReportSummary
    included_state_ids: list[str]
    per_neighbor: list[ReportNeighbor]


class Report(BaseModel):
    """By-generator report for one Crit_gen group. Deliberately has no group-level
    aggregate CCT or stats: a CCT/compactness blended across every fault location within
    the group is exactly the kind of misleading aggregate this model used to have and
    partners misread as a specific answer - per_location carries the real, location-level
    detail, and location_likelihood ranks the locations so callers have a concrete answer
    without needing a blended number."""

    location_likelihood: dict[str, float] = Field(
        ...,
        description=(
            "Location -> score (weight_mass, so it sums to 1.0 across all locations in "
            "this group), sorted descending - the first entry is the single most likely "
            "fault location if a caller just wants one answer, but the full ranking shows "
            "how much more likely it is than the runner-up."
        ),
    )
    per_location: dict[str, LocationReport]
    included_state_ids: list[str]
    neighbors: list[ReportNeighbor]


class LocationGroupReport(BaseModel):
    """By-location report for one fault location - the structural analog of Report (the
    by-generator report) for this endpoint. crit_gen_likelihood ranks critical generators
    within this location, mirroring location_likelihood's job on Report. It cannot reuse
    per_crit_gen's own LocationReport.summary.stats.weight_mass for that ranking, though:
    weight_mass there is normalized within each generator's OWN full neighbor set (so a
    given (location, crit_gen) pair's LocationReport is identical whether reached via
    estimate_by_generator or estimate_by_location - a deliberate invariant), which makes it
    each generator's share of its own group, not a quantity comparable across different
    generators' groups. crit_gen_likelihood is a separate computation from raw (never
    renormalized) query-to-neighbor kernel weights instead, comparable across generators
    for exactly that reason."""

    crit_gen_likelihood: dict[str, float] = Field(
        ...,
        description=(
            "Critical generator -> score (raw query-to-neighbor kernel mass, comparable "
            "across generators since it is never renormalized within any one generator's "
            "own group, then scaled to sum to 1.0 across this dict purely so it reads the "
            "same way as Report.location_likelihood), sorted descending - the first entry "
            "is the single most likely critical generator for this location."
        ),
    )
    per_crit_gen: dict[str, LocationReport]
    included_state_ids: list[str]
    neighbors: list[ReportNeighbor]


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
