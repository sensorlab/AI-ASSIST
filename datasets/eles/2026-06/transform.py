import logging
import re
from pathlib import Path

import click
import joblib
import pandas as pd
from tqdm import tqdm

from scripts._common import write_json_list, write_sqlite_table
from src.config.logging import configure_logging

logger = logging.getLogger(__name__)

RE_PWR = re.compile(r"^(?:U|phi|P\d?|Q\d?|Sk)_", re.IGNORECASE)
RE_OSERV = re.compile(r"^oserv_", re.IGNORECASE)
RE_CCT = re.compile(r"^CCT", re.IGNORECASE)
RE_TYPE = re.compile(r"^Type", re.IGNORECASE)
RE_CAT = re.compile(r"^(?:state|crit_gen|location|terminal)", re.IGNORECASE)
RE_SERIAL = re.compile(r"^experiment", re.IGNORECASE)


def standardize_col_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r"\_$", "", name)  # remove ending underscore
    return name


def filter_topology_cols(lf_cols: list[str]) -> set[str]:
    """oserv_ columns carry topology (in/out-of-service) status directly - no external
    dictionary needed for this drop, unlike eles/2026-01."""
    RE_COLS = re.compile(r"^oserv_", re.IGNORECASE)
    return {c for c in lf_cols if RE_COLS.match(c) is not None}


FSA_METRICS = ("minF", "maxF", "maxRoCoF")


def melt_fsa(fsa: pd.DataFrame, metrics: tuple[str, ...] = FSA_METRICS) -> pd.DataFrame:
    """Reshape wide per-contingency FSA columns (e.g. `minF_<gen1>_<gen2>`) into one row
    per (state, failed_gen, measured_gen). The compound generator-pair suffix isn't a
    numeric suffix, so pd.wide_to_long (used for TSA) doesn't apply - stack per metric and
    concat instead. Every contingency id has exactly one underscore (confirmed empirically:
    generator EIC codes use dashes internally, never underscores), so splitting on it is safe."""
    per_metric = []
    for metric in metrics:
        prefix = f"{metric}_"
        cols = [c for c in fsa.columns if c.startswith(prefix)]
        sub = fsa[cols].copy()
        sub.columns = [c[len(prefix) :] for c in cols]
        series = sub.stack()
        series.name = metric
        per_metric.append(series)

    fsa_long = pd.concat(per_metric, axis=1)
    fsa_long.index.names = ["state", "contingency"]
    fsa_long = fsa_long.reset_index()

    failed_measured = fsa_long.pop("contingency").str.split("_", n=1, expand=True)
    fsa_long["failed_gen"] = failed_measured[0]
    fsa_long["measured_gen"] = failed_measured[1]

    return fsa_long[["state", "failed_gen", "measured_gen", *metrics]]


def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.convert_dtypes(dtype_backend="pyarrow")

    for c in df.columns:
        if RE_PWR.search(c):
            df[c] = df[c].astype("float64[pyarrow]")
        elif RE_OSERV.search(c):
            df[c] = df[c].astype("string[pyarrow]").fillna("1.0").astype(float).astype(int).astype("bool[pyarrow]")
        elif RE_CCT.search(c):
            df[c] = df[c].astype("float64[pyarrow]")
        elif RE_TYPE.search(c):
            df[c] = df[c].astype(float).astype(int).astype("category")
        elif RE_CAT.search(c):
            df[c] = df[c].astype("category")
        elif RE_SERIAL.search(c):
            df[c] = df[c].astype("int64[pyarrow]")
        else:
            logger.error(f"Fallen through {c}\n{df[c].unique()}")
            raise ValueError

    return df


def processor(index: int, in_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Runs in a joblib worker process; see datasets/eles/2026-01/transform.py for why
    # this needs its own logging setup.
    configure_logging()

    lf_csv = in_dir / "LF" / f"LF_main_{index}.csv"
    tsa_csv = in_dir / "TSA" / f"TSA_main_{index}.csv"
    fsa_csv = in_dir / "FSA" / f"FSA_main_{index}.csv"

    if not lf_csv.is_file():
        raise OSError(lf_csv)
    if not tsa_csv.is_file():
        raise OSError(tsa_csv)
    if not fsa_csv.is_file():
        raise OSError(fsa_csv)

    def prepare_lf_dataset() -> pd.DataFrame:
        lf = pd.read_csv(lf_csv, sep=";", decimal=",", index_col=0)
        lf = lf.rename(columns=standardize_col_name)
        lf = lf.copy()

        lf["state"] = f"{index}_" + lf.index.map(str)
        lf = lf.set_index("state")
        if not lf.index.is_unique:
            raise ValueError("Index contains duplicate values")

        return lf

    def prepare_tsa_dataset() -> pd.DataFrame:
        tsa = pd.read_csv(tsa_csv, sep=";", decimal=",", index_col=0)
        tsa = tsa.rename(columns=standardize_col_name)
        tsa = tsa.copy()

        tsa["state"] = f"{index}_" + tsa.index.map(str)

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
        logger.info(f"[{index=}] TSA drop {prev - len(tsa_long)}/{prev} invalid lines")

        tsa_long["Type"] = tsa_long["Type"].apply(float).apply(int)

        tsa_long = optimize_dataframe(tsa_long)
        if not (tsa_long.isnull().sum().sum() == 0):
            raise ValueError(f"Includes invalid values; {index=}")

        return tsa_long

    def prepare_fsa_dataset() -> pd.DataFrame:
        fsa = pd.read_csv(fsa_csv, sep=";", decimal=",", index_col=0)
        fsa = fsa.rename(columns=standardize_col_name)
        fsa = fsa.copy()

        fsa["state"] = f"{index}_" + fsa.index.map(str)
        fsa = fsa.set_index("state")

        fsa_long = melt_fsa(fsa)

        # ~49% of (state, failed_gen, measured_gen) triples have no result: measured_gen was
        # itself out of service in that state, so there's nothing to measure. minF/maxF/
        # maxRoCoF are always null together per triple (confirmed empirically) - drop those
        # rows rather than treating them as invalid.
        prev = len(fsa_long)
        fsa_long = fsa_long.dropna(subset=list(FSA_METRICS), how="all")
        logger.info(f"[{index=}] FSA drop {prev - len(fsa_long)}/{prev} pairs with no result")

        if fsa_long[list(FSA_METRICS)].isnull().sum().sum() != 0:
            raise ValueError(f"FSA includes partially-missing metrics; {index=}")

        return fsa_long

    return prepare_lf_dataset(), prepare_tsa_dataset(), prepare_fsa_dataset()


@click.command()
@click.option(
    "--in-dir",
    "in_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Input directory (extracted clean_files/)",
)
@click.option(
    "--out-dir",
    "out_dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Output directory",
)
def main(in_dir: Path, out_dir: Path):
    """Transform raw eles/2026-06 data into ML-ready pickles."""
    configure_logging()
    out_dir.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(r"LF_main_(?P<idx>\d+)\.csv")
    indices = sorted(int(m.group("idx")) for f in (in_dir / "LF").glob("LF_main_*.csv") if (m := pattern.match(f.name)))

    jobs = [joblib.delayed(processor)(index=index, in_dir=in_dir) for index in indices]
    executor = joblib.Parallel(n_jobs=joblib.cpu_count(), return_as="generator")

    lf_frames: list[pd.DataFrame] = []
    tsa_frames: list[pd.DataFrame] = []
    fsa_frames: list[pd.DataFrame] = []
    for _lf, _tsa, _fsa in tqdm(executor(jobs), total=len(jobs)):
        lf_frames.append(_lf)
        tsa_frames.append(_tsa)
        fsa_frames.append(_fsa)

    lf = pd.concat(lf_frames)
    lf = optimize_dataframe(lf)
    if not lf.index.is_unique:
        raise ValueError("Index contains duplicate values")

    tsa = pd.concat(tsa_frames, ignore_index=True)
    tsa = optimize_dataframe(tsa)

    fsa = pd.concat(fsa_frames, ignore_index=True)
    fsa["state"] = fsa["state"].astype("category")
    fsa["failed_gen"] = fsa["failed_gen"].astype("category")
    fsa["measured_gen"] = fsa["measured_gen"].astype("category")
    for metric in FSA_METRICS:
        fsa[metric] = fsa[metric].astype("float64[pyarrow]")

    logger.debug(f"LF\n{lf.dtypes}")
    logger.debug(f"TSA\n{tsa.dtypes}")
    logger.debug(f"FSA\n{fsa.dtypes}")

    lf.to_pickle(out_dir / "lf.pkl")
    tsa.to_pickle(out_dir / "tsa.pkl")
    fsa.to_pickle(out_dir / "fsa.pkl")

    processed_dir = out_dir.parent / "processed"
    write_sqlite_table(tsa, processed_dir / "tsa.db", table="tsa")
    write_sqlite_table(fsa, processed_dir / "fsa.db", table="fsa")

    topo_cols = filter_topology_cols(list(lf.columns))
    joblib.dump(topo_cols, out_dir / "topology_cols.joblib.z")
    write_json_list(topo_cols, processed_dir / "topology_cols.json")


if __name__ == "__main__":
    main()
