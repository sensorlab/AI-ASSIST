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

    tsa_long = optimize_dataframe(tsa_long)
    # Crit_gen is intentionally NaN for stable faults (CCT = 0.8s ceiling)
    required_cols = [c for c in tsa_long.columns if c != "Crit_gen"]
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
    lf.to_pickle(out_dir / "lf.pkl")

    logger.info("Prepare TSA dataset ...")
    tsa = prepare_tsa_dataset(in_dir / "cct_global_results_clean.csv")
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
