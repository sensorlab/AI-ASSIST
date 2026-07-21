import logging
import multiprocessing as mp
import re
from collections.abc import Iterable
from io import StringIO
from pathlib import Path

import click
import joblib
import pandas as pd
from tqdm import tqdm

from scripts._common import write_json_list, write_sqlite_table
from src.config.logging import configure_logging

logger = logging.getLogger(__name__)

compress_kwargs = {"method": "zstd", "level": 19, "threads": -1}

pd.set_option("compute.use_numba", True)

RE_PWR = re.compile(r"^(?:U|phi|P\d?|Q\d?|Sk)_", re.IGNORECASE)
RE_OSERV = re.compile(r"^oserv_", re.IGNORECASE)
RE_CCT = re.compile(r"^CCT", re.IGNORECASE)
RE_TYPE = re.compile(r"^Type", re.IGNORECASE)
RE_CAT = re.compile(r"^(?:state|crit_gen|location|terminal)", re.IGNORECASE)
RE_SERIAL = re.compile(r"^experiment", re.IGNORECASE)


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


def standardize_col_name(name: str) -> str:
    name = name.strip()
    # name = re.sub(
    #    r"\d+", lambda match: f"{int(match.group(0)):02d}", name
    # )  # convert all single-digits to double digits

    # name = re.sub(r"\_?\[\w+\]", "", name)  # remove units
    # name = re.sub(r"\s+", "_", name)  # change all spaces to underscores
    name = re.sub(r"\_$", "", name)  # remove ending underscore
    # name = name.replace("_-_", "-")  # replace "_-_" with dash
    return name


def filter_slovenian_topology_cols(dict_path: Path, lf_cols: Iterable[str]) -> Iterable[str]:
    RE_COLS = re.compile(r".*for_name.*", flags=re.IGNORECASE)

    xls = pd.ExcelFile(dict_path)
    values: set[str] = set()

    relevant_sheets = [
        # "Generators" # unusable, too many topology changes
        "Loads",
        "Lines",
    ]

    for sheet_name in relevant_sheets:  # xls.sheet_names:
        df = xls.parse(sheet_name)

        # find columns containing the substring
        cols = [c for c in df.columns if RE_COLS.match(str(c))]

        for col in cols:
            # drop NaN to avoid polluting the set
            values.update(df[col].dropna().astype(str))

    values = {standardize_col_name(v) for v in values}

    topo_cols: set[str] = set()
    for col in lf_cols:
        if "oserv_" not in col:
            continue

        for v in values:
            if v in col:
                topo_cols.add(col)
                break

    return topo_cols


FSA_METRICS = ("minF", "maxF", "maxRoCoF", "M1", "M2", "M3")
FSA_THRESHOLD_COLS = ("MThreshold1", "MThreshold2", "MThreshold3")


def melt_fsa(fsa: pd.DataFrame, metrics: tuple[str, ...] = FSA_METRICS) -> pd.DataFrame:
    """Reshape wide per-contingency FSA columns (e.g. `minF_<gen1>_<gen2>`) into one row
    per (state, failed_gen, measured_gen). The compound generator-pair suffix isn't a
    numeric suffix, so pd.wide_to_long (used for TSA) doesn't apply - stack per metric and
    concat instead. Every contingency id has exactly one underscore (confirmed empirically:
    generator EIC codes use dashes internally, never underscores), so splitting on it is
    safe. Mirrors datasets/eles/2026-06/transform.py::melt_fsa."""
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


def get_ignored_lines(path: Path):
    ignores: dict[int, list[int]] = {}

    # pattern in the row ignore list
    ROW_QUERY = r"^\d+_\d+\s*-\s*x+_main_(?P<index>\d+)\.csv,\s*ID\s*=\s*(?P<row>\d+);\s*$"

    with open(path) as fp:
        for row in fp:
            match = re.match(ROW_QUERY, row, re.IGNORECASE)
            if match:
                index = match.group("index")
                row = match.group("row")
                index, row = int(index), int(row)
                ignores.setdefault(index, []).append(row)

    return ignores


_worker_logging_configured = False


def processor(index: int, in_dir: Path, ignore_rows: list[int]) -> tuple[pd.DataFrame, ...]:
    # Runs in a joblib worker process, which doesn't inherit the main process's logging
    # config, so it needs to configure its own timestamped formatting. loky reuses worker
    # processes across jobs, so a module-level flag avoids reconfiguring on every job.
    global _worker_logging_configured
    if not _worker_logging_configured:
        configure_logging()
        _worker_logging_configured = True
    pd.set_option("compute.use_numba", True)

    lf_csv = in_dir / f"LF_main_{index}.csv"
    tsa_csv = in_dir / f"TSA_main_{index}.csv"
    fsa_csv = in_dir / f"FSA_main_{index}.csv"
    sssa_csv = in_dir / f"SSSA_main_{index}.csv"

    if not lf_csv.is_file():
        raise OSError

    if not tsa_csv.is_file():
        raise OSError

    if not fsa_csv.is_file():
        raise OSError

    if not sssa_csv.is_file():
        raise OSError

    def prepare_lf_dataset():
        # Load LF (grid states) records.
        skiprows = [x + 1 for x in ignore_rows]  # offset=1 because index=0 is header

        lf = pd.read_csv(
            lf_csv,
            sep=";",
            decimal=",",
            index_col=0,
            skiprows=skiprows,
        )
        lf = lf.rename(columns=standardize_col_name)
        lf = lf.copy()

        lf["state"] = f"{index}_" + lf.index.map(str)

        lf = lf.set_index("state")
        if not lf.index.is_unique:
            raise ValueError("Index contains duplicate values")

        return lf

    def prepare_tsa_dataset() -> pd.DataFrame:
        text = tsa_csv.read_text()

        # 1. Insert missing separator before decimal values
        fixed = re.sub(r"(?<!;)(\d{1},\d+)(?=;)", r";\1", text)

        # 2. Strip trailing semicolons at end of each line
        fixed = re.sub(r";+$", "", fixed, flags=re.MULTILINE)

        skiprows = [x + 1 for x in ignore_rows]  # offset=1 because index=0 is header

        tsa = pd.read_csv(
            StringIO(fixed),
            sep=";",
            decimal=",",
            index_col=0,
            skiprows=skiprows,
        )
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
        tsa_long = tsa_long.dropna(subset=["Location"], how="all")
        logger.info(
            f"[{index=}] TSA drop {prev - len(tsa_long)}/{prev} entries ({(prev - len(tsa_long)) / prev * 100:.1f}%)"
        )

        def is_missing_like(df: pd.DataFrame, col: str):
            mask = df[col].isna() | df[col].astype(str).str.strip().str.lower().isin(
                {"", "nan", "none", "null", "n/a", "na"}
            )
            if mask.any():
                raise ValueError(f"Column {col} contains missing-like values")

        is_missing_like(tsa_long, "Location")
        is_missing_like(tsa_long, "Terminal")

        # Sanity checks before writing to system.
        if tsa_long.isnull().sum().sum() != 0:
            # print(tsa_long)
            # tsa_long.to_csv(f"TSA_{index}.csv")
            raise ValueError(f"Includes invalid values; {index=}")

        return tsa_long

    def prepare_fsa_dataset() -> pd.DataFrame:
        # Unlike TSA, FSA parses fine as-is - no missing-separator fix needed. (An earlier
        # attempt applied TSA's fix here anyway: it corrupted multi-digit values like the
        # M1/M2/M3 margin metrics' "100,0", e.g. splitting it into "10" and "0,0", because
        # the TSA fix regex only expects a single digit before the decimal comma.)
        skiprows = [x + 1 for x in ignore_rows]  # offset=1 because index=0 is header

        fsa = pd.read_csv(fsa_csv, sep=";", decimal=",", index_col=0, skiprows=skiprows)
        fsa = fsa.rename(columns=standardize_col_name)
        fsa = fsa.copy()

        # MThreshold1/2/3 are global constants (95/97/99% of nominal frequency, confirmed
        # identical across all 89 batches), not per-contingency data - drop rather than melt.
        thresholds = fsa[list(FSA_THRESHOLD_COLS)].drop_duplicates()
        if len(thresholds) != 1:
            raise ValueError(f"Expected constant FSA thresholds within a batch; {index=}, got {thresholds}")
        fsa = fsa.drop(columns=list(FSA_THRESHOLD_COLS))

        fsa["state"] = f"{index}_" + fsa.index.map(str)
        fsa = fsa.set_index("state")

        fsa_long = melt_fsa(fsa)

        # As with eles/2026-06, a large fraction of (state, failed_gen, measured_gen) triples
        # have no result (measured_gen was itself out of service in that state) - all 6
        # metrics are null together for those triples; drop rather than treat as invalid.
        prev = len(fsa_long)
        fsa_long = fsa_long.dropna(subset=list(FSA_METRICS), how="all")
        logger.info(f"[{index=}] FSA drop {prev - len(fsa_long)}/{prev} pairs with no result")

        if fsa_long[list(FSA_METRICS)].isnull().sum().sum() != 0:
            raise ValueError(f"FSA includes partially-missing metrics; {index=}")

        return fsa_long

    lf = prepare_lf_dataset()
    tsa = prepare_tsa_dataset()
    fsa = prepare_fsa_dataset()

    return lf, tsa, fsa

    # Merge LF an TSA datasets (add grid state to each experiment)
    # lf_tsa = tsa.merge(lf, how="left", left_on="state", right_on="state")

    # prev = lf_tsa.shape[1]
    # lf_tsa = lf_tsa.dropna(axis="columns", how="all")
    # print("lf_tsa dropped", prev - lf_tsa.shape[1], "all-NaN columns")

    # ### Process SSSA
    # sssa = pd.read_csv(in_dir / f"SSSA_main_{index}.csv", sep=";", decimal=",", index_col=0)
    # sssa = sssa.rename(columns=standardize_col_name)
    # sssa["state"] = f"{index}_" + sssa.index.map(str)
    # # Not sure what are we estinating here


@click.command()
@click.option(
    "--in-dir",
    "in_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Input directory",
)
@click.option(
    "--ignore-list",
    "ignore_list_path",
    type=click.Path(exists=True, file_okay=True, path_type=Path),
    required=True,
    help="Input directory",
)
@click.option(
    "--dictionary",
    "dictionary_path",
    type=click.Path(exists=True, file_okay=True, path_type=Path),
    required=True,
    help="Dictionary for relevant topology items",
)
@click.option(
    "--out-dir",
    "out_dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Output directory",
)
def main(in_dir: Path, ignore_list_path: Path, dictionary_path: Path, out_dir: Path):
    """Unpack ZIP archive into output directory."""
    configure_logging()

    ignores = get_ignored_lines(ignore_list_path)

    # figure out all indices
    _lf_files = in_dir.glob("LF_main_*.csv")
    pattern = re.compile(r"LF_main_(?P<idx>\d+).csv")

    indices: list[int] = []
    for candidate in _lf_files:
        match = pattern.match(candidate.name)
        if match:
            index = int(match.group("idx"))
            indices.append(index)

    indices.sort()  # ascending order

    jobs = []
    for index in indices:
        ignore_rows = ignores.get(index, [])
        job = joblib.delayed(processor)(index=index, in_dir=in_dir, ignore_rows=ignore_rows)
        jobs.append(job)

    executor = joblib.Parallel(
        # n_jobs=1,
        n_jobs=joblib.cpu_count(),
        return_as="generator",
    )

    lf: list[pd.DataFrame] | pd.DataFrame = []
    tsa: list[pd.DataFrame] | pd.DataFrame = []
    fsa: list[pd.DataFrame] | pd.DataFrame = []
    for _lf, _tsa, _fsa in tqdm(executor(jobs)):
        lf.append(_lf)
        tsa.append(_tsa)
        fsa.append(_fsa)

    lf = pd.concat(lf)
    lf = optimize_dataframe(lf)
    if not lf.index.is_unique:
        raise ValueError("Index contains duplicate values")

    tsa = pd.concat(tsa, ignore_index=True)
    tsa = optimize_dataframe(tsa)

    fsa = pd.concat(fsa, ignore_index=True)
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

    topo_cols = filter_slovenian_topology_cols(dict_path=dictionary_path, lf_cols=lf.columns)
    joblib.dump(topo_cols, out_dir / "topology_cols.joblib.z")
    write_json_list(topo_cols, processed_dir / "topology_cols.json")

    # out: pd.DataFrame = pd.concat(dfs, ignore_index=True)
    # out = optimize_dataframe(out)

    # prev = len(out)
    # out = out.dropna(axis="index")
    # print("out dropped", prev - len(out), "invalid lines")

    # # Sanity checks before writing to system.
    # if out.isnull().sum().sum() != 0:
    #     raise RuntimeError

    # print("Final DataFrame size:", out.shape)
    # # out.to_pickle(out_dir / "lf_tsa_merged.pkl.zst", compression=compress_kwargs)
    # out.to_pickle(out_dir / "lf_tsa_merged.pkl")


if __name__ == "__main__":
    main()
