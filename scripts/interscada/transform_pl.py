import re
from collections.abc import Iterable
from pathlib import Path

import click
import joblib
import numpy as np
import pandas as pd

from src.utils import optimize_dataframe, standardize_col_name


def filter_topology_cols(lf_cols: Iterable[str]) -> set[str]:
    RE_COLS = re.compile(r".*(line_status|gen_status).*", flags=re.IGNORECASE)
    return {c for c in lf_cols if RE_COLS.match(c) is not None}


def prepare_lf_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise OSError

    lf = pd.read_csv(path, sep=";", decimal=".", index_col=0, encoding="utf-8-sig")
    lf = lf.rename(columns=standardize_col_name)
    lf = lf.dropna(axis=1, how="all")  # trailing delimiter may create all-NaN columns
    lf = lf.loc[:, [c for c in lf.columns if str(c).strip()]]  # drop empty/whitespace column names

    # Rename line_status_NN columns to line_status_{from}_{to} using the constant bus-pair values, then drop From/To
    from_cols = [c for c in lf.columns if re.match(r".*line_bus_no_from.*", c, re.IGNORECASE)]
    to_cols = [c for c in lf.columns if re.match(r".*line_bus_no_to.*", c, re.IGNORECASE)]
    stat_cols = [c for c in lf.columns if re.match(r".*line_status.*", c, re.IGNORECASE)]
    bases = [
        f"line_status_{int(lf[f].iloc[0]):02d}_{int(lf[t].iloc[0]):02d}"
        for f, t in zip(from_cols, to_cols, strict=True)
    ]
    counts = {b: bases.count(b) for b in bases}
    seen: dict[str, int] = {}
    rename = {}
    for s, base in zip(stat_cols, bases, strict=True):
        seen[base] = seen.get(base, 0) + 1
        rename[s] = f"{base}_p{seen[base]}" if counts[base] > 1 else base
    lf = lf.drop(columns=from_cols + to_cols).rename(columns=rename)

    # Gen_P_MW columns contain compound " STATUS, P_MW" values (e.g. " 1, 69.45") — split into two columns
    gen_p_cols = [c for c in lf.columns if "gen_p_mw" in c.lower()]
    new_status_cols = {}
    for col in gen_p_cols:
        parts = lf[col].str.split(",")
        new_status_cols[col.lower().replace("gen_p_mw", "gen_status")] = parts.str[0].str.strip().astype(int)
        lf[col] = parts.str[-1].str.strip().astype(float)
    lf = pd.concat([lf, pd.DataFrame(new_status_cols, index=lf.index)], axis=1)

    lf = optimize_dataframe(lf)
    if not (lf.isnull().sum().sum() == 0):
        raise ValueError

    return lf


def prepare_tsa_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise OSError

    tsa = pd.read_csv(path, sep=";", index_col=0)
    tsa = tsa.rename(columns=standardize_col_name)
    tsa["state"] = tsa.index

    # After standardize_col_name: "3PF_location_001" -> "03PF_location_01"
    tsa_long = pd.wide_to_long(
        tsa,
        stubnames=["03PF_location", "CCT", "critical_generator"],
        i=["state"],
        j="experiment",
        sep="_",
        suffix=r"\d+",
    ).reset_index()

    prev = len(tsa_long)
    tsa_long = tsa_long.dropna(subset=["03PF_location", "CCT"], how="all")
    print("tsa_long dropped", prev - len(tsa_long), "invalid lines")

    tsa_long = tsa_long.rename(columns={"03PF_location": "Location"})

    # Stable faults carry string "None" for critical_generator — normalise to NaN
    tsa_long["critical_generator"] = tsa_long["critical_generator"].replace("None", np.nan)
    # Parse PowerFactory element reference e.g. "37 [BUS37       13.8]" -> bus number 37, voltage 13.8
    # Named Crit_gen (not Crit_gen_bus) for compatibility with other datasets, despite being a bus number
    tsa_long["Crit_gen"] = tsa_long["critical_generator"].str.extract(r"^(\d+)").squeeze().astype("Int64")
    tsa_long["voltage_level_kV"] = (
        tsa_long["critical_generator"].str.extract(r"([\d.]+)\]$").squeeze().astype("Float64")
    )
    tsa_long["CCT"] = tsa_long["CCT"].astype(float) / 1000  # ms → seconds

    tsa_long = optimize_dataframe(tsa_long)
    # critical_generator is intentionally NaN for stable faults; exclude it from the null check
    required_cols = [c for c in tsa_long.columns if c not in ("critical_generator", "Crit_gen", "voltage_level_kV")]
    if not (tsa_long[required_cols].isnull().sum().sum() == 0):
        raise ValueError

    return tsa_long


@click.command()
@click.option(
    "--in-dir",
    "in_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Input directory (raw/pl/)",
)
@click.option(
    "--out-dir",
    "out_dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Output directory",
)
def main(in_dir: Path, out_dir: Path):
    """Transform raw interscada PL data into ML-ready pickles."""
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Prepare LF dataset ...")
    lf = prepare_lf_dataset(in_dir / "_power_flow_info_updt.csv")
    lf.to_pickle(out_dir / "lf.pkl")

    print("Prepare TSA dataset ...")
    tsa = prepare_tsa_dataset(in_dir / "_CCT_results_names_ok.csv")
    tsa.to_pickle(out_dir / "tsa.pkl")

    print("Merge LF+TSA dataset ...")
    lf_tsa = tsa.merge(lf, how="left", left_on="state", right_index=True)
    lf_tsa.to_pickle(out_dir / "lf_tsa_merged.pkl")

    # critical_generator(_bus) is intentionally NaN for stable faults; exclude from null check
    required_cols = [c for c in lf_tsa.columns if c not in ("critical_generator", "Crit_gen", "voltage_level_kV")]
    if not (lf_tsa[required_cols].isnull().sum().sum() == 0):
        raise ValueError
    if not (len(lf_tsa) >= len(tsa)):
        raise ValueError

    print("Filter topology cols ...")
    topo_cols = filter_topology_cols(lf_cols=lf.columns)
    joblib.dump(topo_cols, out_dir / "topology_cols.joblib.z")


if __name__ == "__main__":
    main()
