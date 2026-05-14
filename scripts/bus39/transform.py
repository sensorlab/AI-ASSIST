import re
from collections.abc import Iterable
from pathlib import Path
from zipfile import ZipFile

import click
import joblib
import pandas as pd

from src.utils import standardize_col_name

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


def prepare_fsa_dataset(path: Path) -> pd.DataFrame:
    raise NotImplementedError
    # Load FSA

    if not path.exists():
        raise OSError

    fsa = pd.read_csv(path, sep=";", decimal=",", index_col=0)
    fsa = fsa.rename(columns=standardize_col_name)

    # Sanity checks before writing to disk.
    fsa = optimize_dataframe(fsa)
    if not (fsa.isnull().sum().sum() == 0):
        raise ValueError

    return fsa


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
    print("tsa_long dropped", prev - len(tsa_long), "invalid lines")

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
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Prepare LF dataset ...")
    lf = prepare_lf_dataset(in_dir / "LF_main.csv")
    lf.to_pickle(out_dir / "lf.pkl")

    print("Prepare TSA dataset ...")
    tsa = prepare_tsa_dataset(in_dir / "TSA_main.csv")
    tsa.to_pickle(out_dir / "tsa.pkl")

    # print("Prepare FSA dataset ...")
    # fsa = prepare_fsa_dataset(in_dir / "FSA_main.csv")
    # fsa.to_pickle(out_dir / "fsa.pkl")

    print("Merge LF+TSA dataset ...")
    lf_tsa = tsa.merge(lf, how="left", left_on="state", right_index=True)
    lf_tsa.to_pickle(out_dir / "lf_tsa_merged.pkl")

    # Sanity checks before writing to disk.
    if not (lf_tsa.isnull().sum().sum() == 0):
        raise ValueError

    if not (len(lf_tsa) >= len(tsa)):
        raise ValueError

    print("Filter topology cols...")
    topo_cols = filter_topology_cols(lf_cols=lf.columns)
    joblib.dump(topo_cols, out_dir / "topology_cols.joblib.z")

    # print("Merge LF+FSA dataset ...")
    # lf_fsa = fsa.merge(lf, how="left", left_index=True, right_index=True)
    # lf_fsa.to_pickle(out_dir / "lf_fsa_merged.pkl")

    # # Sanity checks before writing to disk.
    # if not (lf_fsa.isnull().sum().sum() == 0):
    #     raise ValueError

    # if not (len(lf_fsa) >= len(fsa)):
    #     raise ValueError


if __name__ == "__main__":
    main()
