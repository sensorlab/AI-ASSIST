import logging
import re
import time
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_TRANSFORM = {
    # "float32": "float64",  # Convert float64 columns to float32
    # "int32": "int64",  # Convert int64 columns to int32
    "category": ["object", "string"],  # Convert object columns (usually strings) to category
}


class TimeIt:
    def __init__(self, msg: str):
        self.msg = msg

    def __enter__(self) -> None:
        self.start = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop = time.perf_counter()
        print(f"{self.msg}: {self.stop - self.start:.3f}s")


def standardize_col_name(name: str) -> str:
    # convert all single-digits to double digits
    name = re.sub(r"\d+", lambda match: f"{int(match.group(0)):02d}", name)
    # name = re.sub(r"\_?\[\w+\]", "", name)  # remove units
    name = re.sub(r"\s+", "_", name)  # change all spaces to underscores
    name = re.sub(r"\_$", "", name)  # remove ending underscore
    name = name.replace("_-_", "-")  # replace "_-_" with dash
    name = name.strip()
    return name


def select_centered_columns(X: pd.DataFrame) -> list[str]:
    columns = X.select_dtypes(include=np.number).columns
    # Select columns that have both negative and positive values
    return [col for col in columns if X[col].min() < 0 and X[col].max() > 0]


def select_positive_columns(X: pd.DataFrame) -> list[str]:
    columns = X.select_dtypes(include=np.number).columns
    # Select columns where the minimum value is >= 0
    return [col for col in columns if X[col].min() >= 0]


def to_numpy(data: Any) -> np.ndarray:
    if isinstance(data, (np.ndarray, list, tuple, set)):
        return np.asarray(data)

    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.to_numpy()

    raise NotImplementedError


def optimize_dataframe(
    df: pd.DataFrame, transform: dict[str, list[str] | str] = DEFAULT_TRANSFORM, inplace: bool = False
) -> pd.DataFrame:
    if not inplace:
        df = df.copy()

    for astype, include in transform.items():
        for col in df.select_dtypes(include=include).columns:
            try:
                df[col] = df[col].astype(astype, errors="raise")
            except Exception as e:
                print(f"Error casting dtypes from '{col}' to {astype} type.\n{e}")
                pass

        # affected_cols = df.select_dtypes(include=include).columns
        # df[affected_cols] = df[affected_cols].astype(astype)

    return df


def determine_features_and_targets(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    # According to the provided documentation only the following variables are relevant:
    feature_cols: list[str] = []
    target_cols: list[str] = []

    features_regex = [
        # from LF_main.csv
        r"^(U|phi)_Bus[_ ]\d+",
        r"^(P|Q)_G[_ ]\d+",
        r"^(P|Q)_Load[_ ]\d+",
        r"^oserv_(Line|G)",
        r"^Sk_Bus[_ ]\d+",
        r"^Sk_of(.*)_at_(.*)",
    ]

    targets_regex = [
        # from TSA_main.csv
        r"^CCT(_\d+)?",
        r"^Terminal(_\d+)?",
        r"^Crit_gen(_\d+)?",
        r"^Type(_\d+)?",
        r"^Location(_\d+)?",
        # From FSA_main.csv
        r"^(min|max)F[_ ]\d+",
        r"^maxRoCoF[_ ]\d+",
        r"^M\d+[_ ]\d+",
    ]

    for column in df.columns:
        column = str(column)

        if re.search("|".join(features_regex), column):
            feature_cols.append(column)
            continue

        if re.search("|".join(targets_regex), column):
            target_cols.append(column)
            continue

        logger.debug(f'Skipped: "{column}"')

    return feature_cols, target_cols
