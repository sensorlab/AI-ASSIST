import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd


class SqliteRecordStore:
    """Indexed-by-state, read-only accessor for one target dataset table (tsa/fsa) written
    once by a dataset's transform.py. Only the rows matching a retrieved neighbor set are ever
    materialized as a DataFrame, so the full table doesn't have to stay resident in every
    service worker process."""

    def __init__(self, path: Path, table: str):
        # `table` is always one of a small set of hardcoded internal literals ("tsa"/"fsa"),
        # never user input - safe to interpolate. `state_ids` in fetch() are the only
        # request-derived values, and those go through parameterized placeholders.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self.table = table
        self.columns = [row[1] for row in self._conn.execute(f"PRAGMA table_info({table})").fetchall()]

    def fetch(self, state_ids: Iterable[Any]) -> pd.DataFrame:
        ids = list(dict.fromkeys(str(s) for s in state_ids))
        if not ids:
            return pd.DataFrame(columns=self.columns)

        placeholders = ",".join("?" * len(ids))
        return pd.read_sql_query(
            f"SELECT * FROM {self.table} WHERE state IN ({placeholders})",
            self._conn,
            params=ids,
        )
