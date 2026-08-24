"""Tests whether exact equality on foreign line-status flags is physically load-bearing.

The deployed compatibility key requires equality on all 907 line-status flags. On this record
none of the 254 Slovenian-specific flags ever changes state, so every one of the 1,785 topology
groups is separated by switching in the neighboring systems carried in the observability-area
model. Methods already argue that generator status is dropped from the key because "that
fragmentation originates outside the system being assessed"; an external review asked why the
same reasoning does not apply to the foreign line flags that are kept.

The asymmetry is defensible on physics - a line outage changes network impedance and power
flows, which is first order for a clearing time, whereas a distant unit's commitment is second
order - but that is an argument, not a measurement. This script measures it.

Mirror of eles_commitment_mismatch.py, which pairs states *within* a line-status group and asks
whether |dCCT| grows with generator-flag disagreement. Here we pair states *across* groups that
share a (location, critical generator) contingency, so the two clearing times describe the same
fault on the same machine, and stratify by how many line flags the pair disagrees on. Stratum 0
is the within-group baseline the deployed key admits; the higher strata are pairs the key
refuses. If |dCCT| at Hamming distance 1 or 2 resembles the baseline, exact equality is
discarding usable support at the margin.

This is an association, not a causal estimate, and it carries the same confound the commitment
analysis does: states differing in few line flags are also likely to be close in time and
dispatch. Continuous feature distance is reported per stratum so that confound is visible
rather than implicit - if |dCCT| tracks feature distance and not Hamming distance, the key is
not what is doing the work.

Run from the repository root:
    uv run python scripts/paper/eles_topology_key_locality.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from src.config.logging import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)

DATASET_DIR: Final[Path] = PROJECT_DIR / "datasets" / "eles" / "2026-06"
PAPER_DATA_DIR: Final[Path] = PROJECT_DIR / "results" / "data"
OUTPUT_PATH: Final[Path] = PAPER_DATA_DIR / "eles_topology_key_locality.csv"
# Pairs sampled per stratum. Low strata are rare, so they are taken whole where fewer exist.
MAX_PAIRS_PER_STRATUM: Final[int] = 4000
STRATA: Final[tuple[tuple[int, int, str], ...]] = (
    (0, 0, "0 (same group, admitted)"),
    (1, 1, "1"),
    (2, 2, "2"),
    (3, 5, "3-5"),
    (6, 10, "6-10"),
    (11, 20, "11-20"),
    (21, 10**6, ">20"),
)
SEED: Final[int] = 42
# Feature-distance-matched comparison: how many admitted/refused pairs to pool, and how many
# equal-count bands of continuous distance to compare them inside.
MATCHED_NONZERO_PAIRS: Final[int] = 60_000
MATCHED_BANDS: Final[int] = 5


def _matched_rows(
    iu: tuple[np.ndarray, np.ndarray],
    ham_u: np.ndarray,
    Z: np.ndarray,
    states: np.ndarray,
    by_state: dict[str, set[tuple[str, str]]],
    outcome: dict[tuple[str, str, str], float],
    rng: np.random.Generator,
) -> list[dict]:
    """Compares admitted and refused pairs *within* bands of continuous feature distance.

    The plain stratification cannot separate the two candidate explanations, because pairs that
    disagree on more line flags are also further apart in the continuous features. Holding
    feature distance roughly fixed and varying only the flag disagreement isolates what exact
    line equality contributes on its own. If the deployed key is doing physical work, refused
    pairs should carry the larger |dCCT| inside a band; if the gradient was only the continuous
    distance that rides along with switching, the two should coincide.
    """
    zero = np.flatnonzero(ham_u == 0)
    nonzero = np.flatnonzero(ham_u > 0)
    nonzero = nonzero[rng.choice(len(nonzero), min(MATCHED_NONZERO_PAIRS, len(nonzero)), replace=False)]
    sel = np.concatenate([zero, nonzero])
    a_idx, b_idx = iu[0][sel], iu[1][sel]
    fdist = np.linalg.norm(Z[a_idx] - Z[b_idx], axis=1)
    admitted = ham_u[sel] == 0

    edges = np.quantile(fdist[admitted], np.linspace(0, 1, MATCHED_BANDS + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    rows: list[dict] = []
    for i in range(MATCHED_BANDS):
        band = (fdist >= edges[i]) & (fdist < edges[i + 1])
        stats_by_arm: dict[str, tuple[float, int, float]] = {}
        for arm, mask in (("admitted (hamming 0)", band & admitted), ("refused (hamming >0)", band & ~admitted)):
            dcct = [
                abs(outcome[(states[a], loc, crit)] - outcome[(states[b], loc, crit)])
                for a, b in zip(a_idx[mask], b_idx[mask], strict=True)
                for loc, crit in by_state.get(states[a], set()) & by_state.get(states[b], set())
            ]
            if dcct:
                stats_by_arm[arm] = (float(np.mean(dcct)), len(dcct), float(np.mean(fdist[mask])))
        if len(stats_by_arm) != 2:
            continue
        (ma, na, fa), (mr, nr, fr) = stats_by_arm["admitted (hamming 0)"], stats_by_arm["refused (hamming >0)"]
        logger.info(
            f"[feature-distance matched] band {i + 1}/{MATCHED_BANDS} "
            f"(dist {fa:.0f} vs {fr:.0f})  admitted {ma:.4f} s (n={na:,})  "
            f"refused {mr:.4f} s (n={nr:,})  refused/admitted {mr / ma:.3f}"
        )
        rows.append(
            {
                "panel": "feature-distance matched",
                "stratum": f"band {i + 1}/{MATCHED_BANDS}",
                "n_matched_contingencies": na + nr,
                "mean_abs_dcct": ma,
                "mean_abs_dcct_refused": mr,
                "mean_feature_distance": fa,
                "mean_feature_distance_refused": fr,
                "refused_over_admitted": mr / ma,
                "n_admitted_contingencies": na,
                "n_refused_contingencies": nr,
            }
        )
    if rows:
        ratios = np.array([r["refused_over_admitted"] for r in rows])
        logger.info(
            f"[feature-distance matched] refused/admitted |dCCT| ratio across bands: "
            f"median {np.median(ratios):.3f}, range {ratios.min():.3f}-{ratios.max():.3f}"
        )
    return rows


def main() -> None:
    configure_logging()
    lf = pd.read_pickle(DATASET_DIR / "interim/lf.pkl")
    lines = [c for c in json.loads((DATASET_DIR / "processed/topology_cols_lines_only.json").read_text()) if c in lf]
    si = set(json.loads((DATASET_DIR / "processed/topology_cols_slovenia_only.json").read_text()))
    varying = [c for c in lines if lf[c].nunique(dropna=False) > 1]
    n_si_varying = sum(c in si for c in varying)
    logger.info(
        f"{len(lines)} line flags in the deployed key; {len(varying)} vary; "
        f"{n_si_varying} of those are Slovenian-specific"
    )
    if n_si_varying:
        logger.warning("A Slovenian-specific flag varies; the premise of this analysis has changed - re-read Methods.")

    flags = lf[varying].astype(np.float32).to_numpy()
    # Hamming on a 0/1 matrix returns the differing fraction; scale back to a flag count.
    ham = np.rint(cdist(flags, flags, metric="hamming") * len(varying)).astype(np.int32)

    # Continuous features, standardized, as the confound control. Topology flags excluded so the
    # two axes stay independent; constant columns dropped so they cannot divide by zero. Columns
    # carrying any NaN are dropped outright rather than imputed: a missing value here means the
    # element is absent from that state, so imputing one would smuggle topology back into the
    # control that is supposed to be independent of it.
    cont = lf.drop(columns=[c for c in lf.columns if c.startswith("oserv_")], errors="ignore")
    cont = cont.select_dtypes(include=[np.number])
    cont = cont.loc[:, cont.notna().all() & (cont.std(numeric_only=True) > 0)]
    Z = ((cont - cont.mean()) / cont.std()).to_numpy(np.float32)
    if not np.isfinite(Z).all():
        raise ValueError("non-finite value survived into the feature-distance control")
    logger.info(f"{Z.shape[1]:,} complete continuous features used for the distance control")

    # Generator-commitment control. The key drops these flags, so pairs at line-Hamming 0 mostly
    # differ in commitment (Methods: 98.80%). Repeating the strata on commitment-identical pairs
    # separates "line status matters" from "states far apart in line status also differ in
    # dispatch", which is the objection the raw stratification cannot answer.
    full_cols = [c for c in json.loads((DATASET_DIR / "processed/topology_cols_full.json").read_text()) if c in lf]
    gen_flags = [c for c in full_cols if c not in set(lines) and lf[c].nunique(dropna=False) > 1]
    gen_mat = lf[gen_flags].astype(np.float32).to_numpy()
    logger.info(f"{len(gen_flags)} varying generator-status flags held fixed in the controlled panel")

    tsa = pd.read_pickle(DATASET_DIR / "interim/tsa.pkl")
    tsa["state"] = tsa["state"].astype(str)
    outcome = {
        (s, loc, gen): cct
        for s, loc, gen, cct in zip(tsa["state"], tsa["Location"], tsa["Crit_gen"], tsa["CCT"], strict=True)
    }
    by_state: dict[str, set[tuple[str, str]]] = {}
    for s, loc, gen in zip(tsa["state"], tsa["Location"], tsa["Crit_gen"], strict=True):
        by_state.setdefault(s, set()).add((loc, gen))

    states = np.array([str(i) for i in lf.index])
    iu = np.triu_indices(len(lf), k=1)
    ham_u = ham[iu]
    rng = np.random.default_rng(SEED)

    # Commitment-identical mask over the same upper-triangle pair ordering.
    gen_ham_u = np.rint(cdist(gen_mat, gen_mat, metric="hamming") * max(len(gen_flags), 1))[iu].astype(np.int32)

    def stratum_row(panel: str, keep: np.ndarray, lo: int, hi: int, label: str) -> dict | None:
        """Mean |dCCT| on contingencies shared by pairs whose line-flag disagreement is in [lo, hi]."""
        sel = np.flatnonzero((ham_u >= lo) & (ham_u <= hi) & keep)
        n_avail = len(sel)
        if n_avail == 0:
            logger.info(f"[{panel}] stratum {label}: no pairs")
            return None
        if n_avail > MAX_PAIRS_PER_STRATUM:
            sel = sel[rng.choice(n_avail, MAX_PAIRS_PER_STRATUM, replace=False)]
        dcct: list[float] = []
        fdist: list[float] = []
        n_shared_pairs = 0
        for a, b in zip(iu[0][sel], iu[1][sel], strict=True):
            shared = by_state.get(states[a], set()) & by_state.get(states[b], set())
            if not shared:
                continue
            n_shared_pairs += 1
            d = float(np.linalg.norm(Z[a] - Z[b]))
            for loc, crit in shared:
                dcct.append(abs(outcome[(states[a], loc, crit)] - outcome[(states[b], loc, crit)]))
                fdist.append(d)
        if not dcct:
            logger.info(f"[{panel}] stratum {label}: {n_avail:,} pairs, none sharing a contingency")
            return None
        arr, fd = np.array(dcct), np.array(fdist)
        logger.info(
            f"[{panel:<20}] hamming {label:<26} pairs {n_avail:>10,}  contingencies {len(arr):>8,}  "
            f"mean |dCCT| {arr.mean():.4f} s  median {np.median(arr):.4f} s  feat.dist {fd.mean():.1f}"
        )
        return {
            "panel": panel,
            "stratum": label,
            "hamming_lo": lo,
            "hamming_hi": hi if hi < 10**6 else "",
            "n_pairs_available": n_avail,
            "n_pairs_sampled": len(sel),
            "n_pairs_sharing_contingency": n_shared_pairs,
            "n_matched_contingencies": len(arr),
            "mean_abs_dcct": float(arr.mean()),
            "median_abs_dcct": float(np.median(arr)),
            "mean_feature_distance": float(fd.mean()),
        }

    panels = (("all pairs", np.ones_like(gen_ham_u, dtype=bool)), ("commitment identical", gen_ham_u == 0))
    rows = [r for panel, keep in panels for lo, hi, lb in STRATA if (r := stratum_row(panel, keep, lo, hi, lb))]
    rows += _matched_rows(iu, ham_u, Z, states, by_state, outcome, rng)

    df = pd.DataFrame(rows)
    for panel in df["panel"].unique():
        m = df["panel"] == panel
        base = df.loc[m & (df["hamming_lo"] == 0), "mean_abs_dcct"]
        if len(base):
            df.loc[m, "excess_over_admitted"] = df.loc[m, "mean_abs_dcct"] / float(base.iloc[0]) - 1.0
        sub = df.loc[m]
        if len(sub) >= 3:
            # Does |dCCT| track the flag count, or only the continuous distance that comes with it?
            r_h = stats.spearmanr(sub["hamming_lo"], sub["mean_abs_dcct"])
            r_f = stats.spearmanr(sub["mean_feature_distance"], sub["mean_abs_dcct"])
            logger.info(
                f"[{panel}] across strata: Spearman(hamming, mean|dCCT|)={r_h.statistic:.3f} "
                f"(p={r_h.pvalue:.3g}); Spearman(feature distance, mean|dCCT|)={r_f.statistic:.3f} "
                f"(p={r_f.pvalue:.3g})"
            )

    PAPER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
