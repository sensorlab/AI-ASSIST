import logging
import re
from collections.abc import Iterable
from pathlib import Path
from zipfile import ZipFile

import click
import joblib
import pandas as pd

from scripts._common import write_json_list, write_sqlite_table
from src.config.logging import configure_logging
from src.utils import standardize_col_name

logger = logging.getLogger(__name__)

# compress_kwargs = {"method": "zstd", "level": 19, "threads": -1}


def filter_topology_cols(lf_cols: Iterable[str]) -> Iterable[str]:
    RE_COLS = re.compile(r"oserve?\_.*", flags=re.IGNORECASE)
    top_cols = {c for c in lf_cols if RE_COLS.match(c) is not None}
    return top_cols


def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.convert_dtypes(dtype_backend="pyarrow")

    for col in df.select_dtypes(include=["str", "object"]).columns:
        df[col] = df[col].astype("category", errors="raise")

    return df


def prepare_lf_dataset(path: Path) -> pd.DataFrame:
    # Load LF (grid states) records.

    if not path.exists():
        raise OSError

    lf = pd.read_csv(path, sep=";", decimal=",", index_col=0)
    lf = lf.rename(columns=standardize_col_name)

    # Quirks: convert everything with `oserv_*` to boolean
    oserv_dtypes = {name: bool for name in lf.columns if name.lower().startswith("oserv")}
    assert len(oserv_dtypes) != 0, oserv_dtypes
    lf = lf.astype(oserv_dtypes, errors="raise")

    lf = optimize_dataframe(lf)
    if not (lf.isnull().sum().sum() == 0):
        raise ValueError

    return lf


FSA_METRICS = ("minF", "maxF", "maxRoCoF", "M1", "M2", "M3")


def prepare_fsa_dataset(path: Path) -> pd.DataFrame:
    """Frequency-stability records. Two quirks from the raw data dictionary
    (Razlaga_zapisa_vzorcev.docx) shape this, both documented in the README's "Unused
    datasets" section:

    1. The dictionary explicitly withholds which physical generator each `_N` index
       corresponds to ("Imena generatorjev niso podane" - generator names are not given),
       while confirming the index is at least consistent across every state (same index
       means the same generator throughout). So `failed_gen` here is deliberately an
       anonymized `fsa_gen_N` label, not comparable to TSA's `Crit_gen` names ("G 03".."G 10")
       even though both range over the same 10 generators.
    2. minF/maxF/maxRoCoF/M1/M2/M3 are global worst-case system values ("Vse veličine so
       gledane globalno, kar pomeni da je predstavljen najslabši primer izmed vseh"), not
       measured at a specific generator - unlike ELES's (failed_gen, measured_gen) pairs,
       there is no measured_gen dimension in this dataset at all. `measured_gen` is a
       constant "system" placeholder purely so this fits EstimationService's shared FSA
       report shape (`_fsa_reports_by_pair()` groups by `["failed_gen", "measured_gen"]`).
    """
    if not path.exists():
        raise OSError

    fsa = pd.read_csv(path, sep=";", decimal=",", index_col=0)
    fsa = fsa.rename(columns=standardize_col_name)
    fsa = fsa.copy()  # defragment
    fsa["state"] = fsa.index

    # standardize_col_name pads every digit run to 2 digits, including the one inside "M1"/
    # "M2"/"M3" themselves (e.g. "M1_0" -> "M01_00") - melt on the padded stub names, then
    # rename back to the data-dictionary-facing M1/M2/M3 for the returned metric columns.
    fsa_long = pd.wide_to_long(
        fsa,
        stubnames=["minF", "maxF", "maxRoCoF", "M01", "M02", "M03"],
        i=["state"],
        j="failed_gen_idx",
        sep="_",
        suffix=r"\d+",
    ).reset_index()
    fsa_long = fsa_long.rename(columns={"M01": "M1", "M02": "M2", "M03": "M3"})

    fsa_long["failed_gen"] = "fsa_gen_" + fsa_long.pop("failed_gen_idx").astype(str)
    fsa_long["measured_gen"] = "system"
    fsa_long = fsa_long[["state", "failed_gen", "measured_gen", *FSA_METRICS]]

    # Quirk: a handful of rows (8/217830 in the current archive, 7 of them failed_gen 0) carry
    # a simulation-divergence artifact from the source tool rather than a real result - e.g.
    # state 15668 has minF_0 = -3.77e+149 verbatim in the raw CSV. maxRoCoF alone catches every
    # currently-known case, but the bound is applied to all three physical metrics as a general
    # "not a plausible frequency-stability result" filter rather than one tuned to today's
    # specific outliers. Bounds are deliberately generous relative to the legitimate range
    # observed in this archive (|minF|/|maxF| <= 1.16, maxRoCoF <= 0.047) so this only catches
    # genuine blow-ups, not real severe events. M1/M2/M3 are margins in seconds with a
    # documented sentinel cap of 100 (no crossing) - never legitimately outside [0, 100].
    prev = len(fsa_long)
    physically_valid = (
        fsa_long["minF"].abs().le(5)
        & fsa_long["maxF"].abs().le(5)
        & fsa_long["maxRoCoF"].abs().le(5)
        & fsa_long["M1"].between(0, 100)
        & fsa_long["M2"].between(0, 100)
        & fsa_long["M3"].between(0, 100)
    )
    fsa_long = fsa_long.loc[physically_valid]
    logger.info(f"fsa_long dropped {prev - len(fsa_long)} rows with implausible (simulation-divergence) values")

    # Sanity checks before writing to disk.
    fsa_long = optimize_dataframe(fsa_long)
    if not (fsa_long.isnull().sum().sum() == 0):
        raise ValueError

    return fsa_long


def prepare_tsa_dataset(path: Path) -> pd.DataFrame:
    # Load TSA (transient states) records.

    if not path.exists():
        raise OSError

    tsa = pd.read_csv(path, sep=";", decimal=",", index_col=0, engine="python")
    tsa = tsa.copy()  # defragment
    tsa = tsa.rename(columns=standardize_col_name)
    tsa["state"] = tsa.index

    # Convert TSA from wide to long format.
    tsa_long = pd.wide_to_long(
        tsa,
        stubnames=["Type", "Crit_gen", "Location", "Terminal", "CCT"],
        i=["state"],
        j="experiment",
        sep="_",
        suffix=r"\d+",
    ).reset_index()

    # Drop invalid row where there is no type or location of grid issue.
    prev = len(tsa_long)
    tsa_long = tsa_long.dropna(subset=["Type", "Location"], how="all")
    logger.info(f"tsa_long dropped {prev - len(tsa_long)} invalid lines")

    # Quirks: Some values in `Type` are float, some are string of floats
    tsa_long["Type"] = tsa_long["Type"].apply(float).apply(int)

    # Sanity checks before writing to system.
    tsa_long = optimize_dataframe(tsa_long)
    if not (tsa_long.isnull().sum().sum() == 0):
        raise ValueError

    return tsa_long


@click.command()
@click.option(
    "--in-dir",
    "in_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Input directory",
)
@click.option(
    "--out-dir",
    "out_dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Output directory",
)
def main(in_dir: Path, out_dir: Path):
    """Unpack ZIP archive into output directory."""
    configure_logging()
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Prepare LF dataset ...")
    lf = prepare_lf_dataset(in_dir / "LF_main.csv")
    lf.to_pickle(out_dir / "lf.pkl")

    logger.info("Prepare TSA dataset ...")
    tsa = prepare_tsa_dataset(in_dir / "TSA_main.csv")
    tsa.to_pickle(out_dir / "tsa.pkl")
    write_sqlite_table(tsa, out_dir.parent / "processed" / "tsa.db", table="tsa")

    logger.info("Prepare FSA dataset ...")
    fsa = prepare_fsa_dataset(in_dir / "FSA_main.csv")
    fsa.to_pickle(out_dir / "fsa.pkl")
    write_sqlite_table(fsa, out_dir.parent / "processed" / "fsa.db", table="fsa")

    logger.info("Merge LF+TSA dataset ...")
    lf_tsa = tsa.merge(lf, how="left", left_on="state", right_index=True)
    lf_tsa.to_pickle(out_dir / "lf_tsa_merged.pkl")

    # Sanity checks before writing to disk.
    if not (lf_tsa.isnull().sum().sum() == 0):
        raise ValueError

    if not (len(lf_tsa) >= len(tsa)):
        raise ValueError

    logger.info("Filter topology cols...")
    topo_cols = filter_topology_cols(lf_cols=lf.columns)
    joblib.dump(topo_cols, out_dir / "topology_cols.joblib.z")
    write_json_list(topo_cols, out_dir.parent / "processed" / "topology_cols.json")


if __name__ == "__main__":
    main()
