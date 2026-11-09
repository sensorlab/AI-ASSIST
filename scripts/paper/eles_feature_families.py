"""Which feature families carry the ELES distance? (exploratory analysis 11, R2.10)

Recorded in the manuscript repository's revision/analyses/11-feature-families.md before any number here was computed.
Exploratory: no decision rule, and the deployed representation stays. Each family of _representations.FAMILIES is
removed from the distance (without_<f>) and used alone (only_<f>), through the same harness, pairing and intervals
as analysis 05 (eles_representations.main), read against the same 0.010 s criterion.

Run from the repository root:
    nice -n 19 uv run python scripts/paper/eles_feature_families.py [n_jobs]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _representations  # noqa: E402
from eles_representations import main  # noqa: E402

OUTPUT_PATH: Final[Path] = Path(__file__).resolve().parents[2] / "results" / "data" / "eles_feature_families.csv"

if __name__ == "__main__":
    main(_representations.FAMILY_NAMES, OUTPUT_PATH)
