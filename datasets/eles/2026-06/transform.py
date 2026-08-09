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


def filter_topology_cols_full(lf_cols: list[str]) -> set[str]:
    """Every oserv_ column (both oserv_Gen* and oserv_Lne*) - no external dictionary needed
    for this drop, unlike eles/2026-01. Fragments almost every record into its own singleton
    topology group in practice (~95% singleton on the 2026-07-25 investigation) - kept for
    continuity/comparison, not recommended. See datasets/eles/2026-06/README.md's "Topology
    Variants" section."""
    RE_COLS = re.compile(r"^oserv_", re.IGNORECASE)
    return {c for c in lf_cols if RE_COLS.match(c) is not None}


def filter_topology_cols_lines_only(lf_cols: list[str]) -> set[str]:
    """All oserv_Lne* columns, oserv_Gen* excluded. Generator commitment is a dispatch
    decision, not a network topology change; oserv_Gen* is dropped from the hard equality key
    to reduce fragmentation, on the empirical basis that "generator off" is recoverable from
    the continuous P_Gen*/Q_Gen* power features (nonzero iff "on" in every generator checked -
    see the README section below for the caveat on how much weight that observation can bear,
    and note it establishes only on/off recoverability, not that every retrieval-relevant
    aspect of generator commitment survives the drop). This coverage/compatibility tradeoff is
    the deployed default: on the current 4,393-state artifact it gives 1,785 topology groups
    with 76.1% of records having >=1 same-group neighbor, versus either near-total
    fragmentation ("full") or total collapse ("slovenia_only", on this data)."""
    RE_COLS = re.compile(r"^oserv_Lne", re.IGNORECASE)
    return {c for c in lf_cols if RE_COLS.match(c) is not None}


def filter_topology_cols_slovenia_only(dict_path: Path, lf_cols: list[str]) -> set[str]:
    """Dictionary-matched subset (ELES's own PowerFactory `Lines`+`Loads` sheets, generators
    excluded) - identical matching logic to eles/2026-01's filter_slovenian_topology_cols(),
    reused here against eles/2026-01's dictionary since this drop has no dictionary of its
    own and the underlying element names are confirmed identical between the two drops (see
    the README section below). Degenerate (collapses to 1 topology group) on the data
    available as of 2026-07-25 - not a flaw in this matching logic, but a reflection of the
    current, still-growing data batch (no recorded Slovenian topology change yet). Kept and
    should be re-evaluated as more simulation batches arrive, not discarded."""
    RE_COLS = re.compile(r".*for_name.*", flags=re.IGNORECASE)

    xls = pd.ExcelFile(dict_path)
    values: set[str] = set()

    relevant_sheets = [
        # "Generators" # unusable, too many topology changes
        "Loads",
        "Lines",
    ]

    for sheet_name in relevant_sheets:
        df = xls.parse(sheet_name)
        cols = [c for c in df.columns if RE_COLS.match(str(c))]
        for col in cols:
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


SSSA_MODE_RE = re.compile(r"^(RealPart|ImagPart)_Mode(\d+)$")
# Unlike eles/2026-01, this dataset's SSSA export does break each metric down per generator
# state variable (speed/phi/flux linkages) - confirmed empirically, and ConMag/ConAng are
# absent here (present in eles/2026-01) - PowerFactory's SSSA report evidently ran with
# different settings for the two data drops.
SSSA_PARTICIPATION_RE = re.compile(
    r"^(?P<metric>ConMag|ConAng|ObsMag|ObsAng|ParMag|ParAng)"
    r"_Mode(?P<mode>\d+)_(?P<state_variable>speed|phi|Psi1d|Psifd|Psi1q|Psi2q)_(?P<generator>.+)$"
)


def normalize_sssa_generator(name: str) -> str:
    """Clean up a stray underscore some SSSA generator names have before a dash (e.g.
    "NEK_-G1" -> "NEK-G1"). Does not map to TSA/FSA's EIC codes - that mapping only covers
    named plants (see datasets/eles/2026-01/raw/powerfactory_dictionary.xlsx's Generators
    sheet) and is deliberately deferred until SSSA has an actual consumer."""
    return re.sub(r"_-", "-", name)


def melt_sssa_modes(sssa: pd.DataFrame) -> pd.DataFrame:
    """Reshape the RealPart_Mode{N}/ImagPart_Mode{N} eigenvalue columns (shared shape with
    eles/2026-01) into one row per (state, mode_id).

    IMPORTANT: mode_id is only unique within one state - per the data dictionary, "Oscillatory
    modes of different operating points with the same names, aren't necessarily the same
    oscillatory modes". Never compare, join, or aggregate mode_id across different states -
    named "mode_id" rather than "mode" specifically to make that misuse harder to reach for."""
    mode_cols = [c for c in sssa.columns if SSSA_MODE_RE.match(c)]
    long = pd.wide_to_long(
        sssa[["state", *mode_cols]],
        stubnames=["RealPart_Mode", "ImagPart_Mode"],
        i="state",
        j="mode_id",
        sep="",
        suffix=r"\d+",
    ).reset_index()
    return long.rename(columns={"RealPart_Mode": "real_part", "ImagPart_Mode": "imag_part"})


def melt_sssa_participation(sssa: pd.DataFrame) -> pd.DataFrame:
    """Reshape the per-(mode_id, state_variable, generator) participation/observability
    columns into one row per (state, mode_id, generator), with e.g. ObsMag_speed/ParAng_Psi2q
    as columns. Missingness across state variables is structurally meaningful (tracks
    generator model fidelity, not noise) - no dropna here. Mirrors
    datasets/eles/2026-01/transform.py::melt_sssa_participation structurally, but that
    version has no state_variable dimension to fold into the column name.

    IMPORTANT: mode_id is only unique within one state - see melt_sssa_modes()."""
    gen_cols = [c for c in sssa.columns if c != "state" and not SSSA_MODE_RE.match(c)]

    tuples = []
    for col in gen_cols:
        m = SSSA_PARTICIPATION_RE.match(col)
        if not m:
            raise ValueError(f"Unrecognized SSSA column: {col!r}")
        generator = normalize_sssa_generator(m["generator"])
        tuples.append((int(m["mode"]), generator, f"{m['metric']}_{m['state_variable']}"))

    sub = sssa.set_index("state")[gen_cols].copy()
    sub.columns = pd.MultiIndex.from_tuples(tuples, names=["mode_id", "generator", "metric"])
    return sub.stack(["mode_id", "generator"], future_stack=True).reset_index()


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


def sssa_processor(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Unlike LF/TSA/FSA (batched, many states per file), this dataset's SSSA is one file
    per state under clean_files/SSSA/<timestamp>_SSSA.csv - so this runs once per file
    rather than once per batch index, with its own discovery loop in main()."""
    configure_logging()

    state = path.name.removesuffix("_SSSA.csv")
    sssa = pd.read_csv(path, sep=";", decimal=",")
    sssa = sssa.rename(columns=standardize_col_name)
    # Raw <YYYYMMDD>_<HHMM> timestamp - not yet the {batch}_{row} state id everything else
    # uses (LF/TSA/FSA). main() remaps this via _load_sssa_state_mapping() after collecting
    # all files, since a single file has no way to know its own (batch, row) on its own.
    sssa["state"] = state

    return melt_sssa_modes(sssa), melt_sssa_participation(sssa)


def _load_sssa_state_mapping(in_dir: Path) -> dict[str, str]:
    """SSSA files are keyed by <YYYYMMDD>_<HHMM> timestamp (the SSSA_processor's raw
    `state`), but LF/TSA/FSA use the {batch}_{row} state id. clean_files/Dates/Date_main_N.csv
    gives the DateTime for each row of batch N (its own row index matches LF_main_N.csv's row
    index for that same batch) - confirmed empirically: Date_main_0.csv has 48 rows, one
    DateTime per LF_main_0.csv row, filename-matching exactly the corresponding SSSA file.
    This builds the reverse mapping (timestamp -> {batch}_{row}) so SSSA rows can be
    translated onto the same state id everything else uses."""
    pattern = re.compile(r"Date_main_(?P<idx>\d+)\.csv")
    mapping: dict[str, str] = {}
    for path in sorted((in_dir / "Dates").glob("Date_main_*.csv")):
        m = pattern.match(path.name)
        if not m:
            continue
        batch = int(m["idx"])
        dates = pd.read_csv(path, sep=";", index_col=0)
        for row, timestamp in dates["DateTime"].items():
            state = f"{batch}_{row}"
            existing = mapping.get(timestamp)
            if existing is not None and existing != state:
                raise ValueError(f"Timestamp {timestamp!r} maps to both {existing!r} and {state!r}")
            mapping[timestamp] = state
    return mapping


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

    # SSSA is one file per state here (not batched like LF/TSA/FSA above), so it gets its
    # own discovery loop rather than reusing the `indices` list.
    sssa_files = sorted((in_dir / "SSSA").glob("*_SSSA.csv"))
    sssa_jobs = [joblib.delayed(sssa_processor)(path=path) for path in sssa_files]
    sssa_modes_frames: list[pd.DataFrame] = []
    sssa_participation_frames: list[pd.DataFrame] = []
    for _sssa_modes, _sssa_participation in tqdm(executor(sssa_jobs), total=len(sssa_jobs)):
        sssa_modes_frames.append(_sssa_modes)
        sssa_participation_frames.append(_sssa_participation)

    sssa_modes = pd.concat(sssa_modes_frames, ignore_index=True)
    sssa_participation = pd.concat(sssa_participation_frames, ignore_index=True)

    # Translate SSSA's raw <YYYYMMDD>_<HHMM> timestamps onto the {batch}_{row} state id
    # LF/TSA/FSA use - without this, sssa's "state" values never match anything else's. A
    # handful of SSSA timestamps have no corresponding Dates/LF row at all (confirmed: e.g.
    # 20240627_0915 falls in a gap where Date_main_0.csv skips straight from 0815 to 1215) -
    # such states can never be retrieved as an LF neighbor anyway, so they're dropped rather
    # than treated as an error, same as other legitimate SSSA/FSA coverage gaps.
    state_mapping = _load_sssa_state_mapping(in_dir)
    sssa_modes["state"] = sssa_modes["state"].map(state_mapping)
    sssa_participation["state"] = sssa_participation["state"].map(state_mapping)

    prev = len(sssa_modes)
    sssa_modes = sssa_modes.dropna(subset=["state"])
    logger.info(f"SSSA modes drop {prev - len(sssa_modes)}/{prev} rows with no matching LF state")

    prev = len(sssa_participation)
    sssa_participation = sssa_participation.dropna(subset=["state"])
    logger.info(f"SSSA participation drop {prev - len(sssa_participation)}/{prev} rows with no matching LF state")

    # real_part/imag_part are mode-level values that lived in the same raw CSV row as every
    # per-generator participation column before melt_sssa_participation() exploded that row
    # into one line per generator - this join just puts them back next to the participation
    # values they originally sat beside. They're expected to repeat across every generator row
    # sharing a (state, mode_id), not an artifact of the join.
    sssa = sssa_participation.merge(sssa_modes, on=["state", "mode_id"], how="inner")

    logger.debug(f"LF\n{lf.dtypes}")
    logger.debug(f"TSA\n{tsa.dtypes}")
    logger.debug(f"FSA\n{fsa.dtypes}")
    logger.debug(f"SSSA\n{sssa.dtypes}")

    lf.to_pickle(out_dir / "lf.pkl")
    tsa.to_pickle(out_dir / "tsa.pkl")
    fsa.to_pickle(out_dir / "fsa.pkl")
    sssa.to_pickle(out_dir / "sssa.pkl")

    processed_dir = out_dir.parent / "processed"
    write_sqlite_table(tsa, processed_dir / "tsa.db", table="tsa")
    write_sqlite_table(fsa, processed_dir / "fsa.db", table="fsa")
    write_sqlite_table(sssa, processed_dir / "sssa.db", table="sssa")

    # Three named topology-column variants (see README's "Topology Variants" section for the
    # full investigation behind this) - written as topology_cols_<variant>.json so
    # EstimationService can select one via config rather than there being a single hardcoded
    # definition. "lines_only" is the deployed default going forward - chosen for coverage
    # after inspecting all three variants on this data, not validated against point accuracy.
    # out_dir is datasets/eles/2026-06/interim; walk up to datasets/eles/ then into 2026-01's
    # raw/ - this dataset has no dictionary of its own (see filter_topology_cols_slovenia_only).
    dict_path = out_dir.parent.parent / "2026-01" / "raw" / "powerfactory_dictionary.xlsx"
    variants = {
        "full": filter_topology_cols_full(list(lf.columns)),
        "lines_only": filter_topology_cols_lines_only(list(lf.columns)),
        "slovenia_only": filter_topology_cols_slovenia_only(dict_path, list(lf.columns)),
    }
    for variant_name, topo_cols in variants.items():
        joblib.dump(topo_cols, out_dir / f"topology_cols_{variant_name}.joblib.z")
        write_json_list(topo_cols, processed_dir / f"topology_cols_{variant_name}.json")
        logger.info(f"Topology variant {variant_name!r}: {len(topo_cols)} columns")


if __name__ == "__main__":
    main()
