import pandas as pd

from src import DATA_INTERIM_PATH, DATA_RAW_PATH
from src.utils import optimize_dataframe, standardize_col_name

INPUT_PATH = DATA_RAW_PATH / "Results_v2" / "main"
assert INPUT_PATH.exists(), INPUT_PATH

OUTPUT_PATH = DATA_INTERIM_PATH
assert OUTPUT_PATH.exists(), OUTPUT_PATH

compress_kwargs = {"method": "zstd", "level": 19, "threads": -1}


def main():
    # Load LF (grid states) records.
    lf = pd.read_csv(INPUT_PATH / "LF_main.csv", sep=";", decimal=",", index_col=0)
    lf = lf.rename(columns=standardize_col_name)

    # lf = lf.convert_dtypes()
    lf = optimize_dataframe(lf)

    assert lf.isnull().sum().sum() == 0

    # Quirks: convert everything with `oserv_*` to boolean
    oserv_dtypes = {name: bool for name in lf.columns if name.lower().startswith("oserv")}
    assert len(oserv_dtypes) != 0, oserv_dtypes
    lf = lf.astype(oserv_dtypes)

    lf.to_pickle(OUTPUT_PATH / "lf.pkl.zst", compression=compress_kwargs)

    # Load TSA (transient states) records.
    tsa = pd.read_csv(INPUT_PATH / "TSA_main.csv", sep=";", decimal=",", index_col=0)
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

    # Optimize types and size
    tsa_long = optimize_dataframe(tsa_long)

    # Sanity checks before writing to system.
    assert tsa_long.isnull().sum().sum() == 0

    tsa_long.to_pickle(OUTPUT_PATH / "tsa.pkl.zst", compression=compress_kwargs)

    # Merge LF an TSA datasets (add grid state to each experiment)
    df = tsa_long.merge(lf, how="left", left_on="state", right_index=True)

    # Optimize datatypes
    # df = df.convert_dtypes()
    df = optimize_dataframe(df)

    # Sanity checks before writing to disk.
    assert df.isnull().sum().sum() == 0

    df.to_pickle(OUTPUT_PATH / "tsa_lf_merged.pkl.zst", compression=compress_kwargs)

    ### FSA ###
    fsa = pd.read_csv(INPUT_PATH / "FSA_main.csv", sep=";", decimal=",", index_col=0)
    fsa = fsa.rename(columns=standardize_col_name)
    # fsa = optimize_dataframe(fsa)

    # Sanity checks before writing to disk.
    assert fsa.isnull().sum().sum() == 0

    fsa.to_pickle(OUTPUT_PATH / "fsa.pkl.zst", compression=compress_kwargs)

    # Merge LF an TSA datasets (add grid state to each experiment)
    df = fsa.merge(lf, how="left", left_index=True, right_index=True)

    # Sanity checks before writing to disk.
    assert df.isnull().sum().sum() == 0
    assert len(df) >= len(fsa)

    df.to_pickle(OUTPUT_PATH / "fsa_lf_merged.pkl.zst", compression=compress_kwargs)


if __name__ == "__main__":
    main()
