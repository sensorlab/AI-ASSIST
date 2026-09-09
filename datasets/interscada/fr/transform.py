import logging
import re
from pathlib import Path

import click
import joblib
import pandas as pd

from scripts._common import write_json_list, write_sqlite_table
from src.config.logging import configure_logging
from src.utils import optimize_dataframe

logger = logging.getLogger(__name__)


def filter_topology_cols(line_cols) -> set[str]:
    return set(line_cols)


def prepare_lf_dataset(static_path: Path, line_path: Path) -> tuple[pd.DataFrame, set[str]]:
    if not static_path.exists():
        raise OSError(static_path)
    if not line_path.exists():
        raise OSError(line_path)

    static = pd.read_csv(static_path, sep=";", index_col=0)
    line = pd.read_csv(line_path, sep=";", index_col=0)

    topo_cols = filter_topology_cols(line.columns)

    lf = pd.concat([static, line], axis=1)
    lf = optimize_dataframe(lf)

    return lf, topo_cols


def prepare_tsa_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise OSError(path)

    tsa = pd.read_csv(path, sep=",", index_col=0)

    # Columns alternate: (CCT, Crit_gen, CCT, Crit_gen, ...)
    # First CCT column header is "Fault – NAME", subsequent ones are just "NAME"
    cct_cols = tsa.columns[0::2]
    gen_cols = tsa.columns[1::2]

    tsa["state"] = tsa.index

    frames = []
    for cct_col, gen_col in zip(cct_cols, gen_cols, strict=True):
        fault = re.sub(r"^Fault\s*[–-]\s*", "", cct_col).strip()
        frame = tsa[["state", cct_col, gen_col]].copy()
        frame.columns = ["state", "CCT", "Crit_gen"]
        frame["Location"] = fault
        frames.append(frame)

    tsa_long = pd.concat(frames, ignore_index=True)
    # File stores t_clearance with fault inserted at t=1s, so actual CCT = value - 1.0s
    tsa_long["CCT"] = pd.to_numeric(tsa_long["CCT"], errors="coerce") - 1.0
    # Drop CALC PB rows — simulation failed, no usable target
    prev = len(tsa_long)
    tsa_long = tsa_long.dropna(subset=["CCT"])
    logger.info(f"tsa_long dropped {prev - len(tsa_long)} CALC PB rows")

    # A fault here takes a whole group of machines out of synchronism, so the raw cell lists
    # them comma-separated (e.g. "PALUE7PALUET1, PALUE7PALUET3, PALUE7PALUET4"). Kept whole, the
    # three distinct strings in this drop would act as three opaque classes, two of them nested,
    # and Crit_gen would mean something different here than in every other dataset. Exploding to
    # one row per machine restores the project-wide meaning of Crit_gen as a single generator,
    # and the machine names then match the pilot's six.
    #
    # The full group is kept in crit_gen_group, because the explosion is lossy: from the
    # per-machine rows alone there is no way to tell which machines lost synchronism together in
    # one event. CCT is replicated across a group's rows - it is a property of the fault, not of
    # the individual machine - so any count of records over-counts events by the group size.
    tsa_long["crit_gen_group"] = tsa_long["Crit_gen"]
    tsa_long["Crit_gen"] = tsa_long["Crit_gen"].map(
        lambda value: [part.strip() for part in str(value).split(",")] if pd.notna(value) else [value]
    )
    prev = len(tsa_long)
    tsa_long = tsa_long.explode("Crit_gen", ignore_index=True)
    logger.info(
        f"Exploded multi-generator outcomes: {prev} rows -> {len(tsa_long)} "
        f"({tsa_long['Crit_gen'].nunique()} distinct critical generators)"
    )

    tsa_long = optimize_dataframe(tsa_long)
    # Crit_gen and crit_gen_group are intentionally NaN for stable faults (CCT at the ceiling)
    required_cols = [c for c in tsa_long.columns if c not in ("Crit_gen", "crit_gen_group")]
    if not (tsa_long[required_cols].isnull().sum().sum() == 0):
        raise ValueError

    return tsa_long


@click.command()
@click.option(
    "--in-dir",
    "in_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Input directory (raw/fr/)",
)
@click.option(
    "--out-dir",
    "out_dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Output directory",
)
def main(in_dir: Path, out_dir: Path):
    """Transform raw interscada FR data into ML-ready pickles."""
    configure_logging()
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Prepare LF dataset ...")
    lf, topo_cols = prepare_lf_dataset(
        in_dir / "static_global_results_clean.csv",
        in_dir / "line_global_results_clean.csv",
    )

    logger.info("Prepare TSA dataset ...")
    tsa = prepare_tsa_dataset(in_dir / "cct_global_results_clean.csv")

    # Exact duplicate scenarios. The September 2026 drop stores 264 of its 1,571 scenarios as
    # byte-identical copies of another scenario - 240 pairs plus one group of 25 - and the
    # copies carry identical outcomes too (all 453 duplicate pairs agree on CCT at every shared
    # fault location), so they add no information. They are not merely a nuisance for
    # evaluation, where a query would retrieve a verbatim copy of its own answer despite
    # leave-one-state-out: a duplicated operating point also gets counted once per copy in the
    # kernel-weighted mean, so 25 copies of one state would carry 25 times the weight of a
    # state recorded once. Detected by exact row equality rather than a distance threshold,
    # which needs no tolerance and does not change with float precision.
    before = len(lf)
    lf = lf[~lf.duplicated(keep="first")]
    if before != len(lf):
        logger.info(f"Dropping {before - len(lf)} duplicate LF states, {len(lf)} distinct states retained")

    # The static/line/hvdc files and the CCT file do not always cover the same scenarios: the
    # September 2026 drop carries 1,571 in the former and 1,568 in the latter. A state with no
    # simulated outcome cannot support an estimate, and EstimationService treats a retrieved
    # state with no TSA record as an error rather than a silent gap, so such states are dropped
    # from the reference library here rather than being indexed and failing at query time.
    orphans = lf.index.difference(pd.Index(tsa["state"].unique()))
    if len(orphans):
        logger.info(f"Dropping {len(orphans)} LF states with no TSA record: {sorted(orphans)}")
        lf = lf.loc[~lf.index.isin(orphans)]

    # Keep the outcome table aligned with the retained reference library, so record counts and
    # coverage denominators refer to the same set of states everything else does.
    prev = len(tsa)
    tsa = tsa[tsa["state"].isin(set(lf.index.astype(str)))]
    if prev != len(tsa):
        logger.info(f"Dropped {prev - len(tsa)} TSA rows belonging to removed states")

    lf.to_pickle(out_dir / "lf.pkl")
    tsa.to_pickle(out_dir / "tsa.pkl")
    write_sqlite_table(tsa, out_dir.parent / "processed" / "tsa.db", table="tsa")

    logger.info("Merge LF+TSA dataset ...")
    lf_tsa = tsa.merge(lf, how="left", left_on="state", right_index=True)
    lf_tsa.to_pickle(out_dir / "lf_tsa_merged.pkl")

    # Static file has sparse bus data (n_buses varies per scenario) — NaN in LF cols is expected.
    # Only assert completeness on the TSA-derived columns.
    if not (lf_tsa[["state", "Location", "CCT"]].isnull().sum().sum() == 0):
        raise ValueError
    if not (len(lf_tsa) >= len(tsa)):
        raise ValueError

    logger.info("Filter topology cols ...")
    joblib.dump(topo_cols, out_dir / "topology_cols.joblib.z")
    write_json_list(topo_cols, out_dir.parent / "processed" / "topology_cols.json")


if __name__ == "__main__":
    main()
