from typing import Literal

from pydantic import BaseModel


class ReportSummary(BaseModel):
    cct_weighted: float
    cct_weighted_per_location: dict[str, float]
    location_weight_mass: dict[str, float]
    neighborhood_density: float
    n: int
    n_eff: float
    distances: dict[str, float]
    location_counts: dict[str, int]


class ReportNeighbor(BaseModel):
    state: str
    cct: float
    location: str
    terminal: str
    type: Literal[0, 1]
    weight: float


class Report(BaseModel):
    summary: ReportSummary
    included_state_ids: list[str]
    per_neighbor: list[ReportNeighbor]
