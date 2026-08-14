"""Shared helpers for the scripts/*/prepare*.py dataset-preparation scripts."""

import json
import logging
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent

_ELES_DATE_FILE_RE = re.compile(r"Date_main_(?P<idx>\d+)\.csv")


def load_eles_state_timestamps(zip_path: Path) -> pd.Series:
    """Map each ELES state id to its acquisition timestamp, read from the raw DSA archive.

    Uses the same {batch}_{row} state-id construction as
    datasets/eles/*/transform.py::_load_sssa_state_mapping(), inverted (state -> timestamp).
    Only the Dates/ members are extracted, into a TemporaryDirectory that is removed on exit
    regardless of success, so no copy of the operational timestamps is left on disk.

    The raw ZIP is the only persistent location for these timestamps - they are deliberately
    not carried into interim/ or processed/ - so any analysis needing state chronology reads
    them through here rather than re-implementing the join.
    """
    rows: list[tuple[str, str]] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        with zipfile.ZipFile(zip_path) as zf:
            members = [m for m in zf.namelist() if "/Dates/" in m or m.startswith("Dates/")]
            zf.extractall(tmp_dir, members=members)
        for path in sorted(tmp_dir.glob("**/Date_main_*.csv")):
            match = _ELES_DATE_FILE_RE.match(path.name)
            if not match:
                continue
            batch = int(match["idx"])
            dates = pd.read_csv(path, sep=";", index_col=0)
            for row, timestamp in dates["DateTime"].items():
                rows.append((f"{batch}_{row}", timestamp))
    state_ts = pd.DataFrame(rows, columns=["state", "timestamp"]).set_index("state")
    return pd.to_datetime(state_ts["timestamp"], format="%Y%m%d_%H%M")


def run_script(script: Path, *args: str) -> None:
    command = [sys.executable, str(script), *args]
    logger.debug(f"Running: {' '.join(command)}")
    subprocess.run(command, check=True, cwd=REPO_ROOT)


def extract_zip(zip_path: Path, out_dir: Path, junk_paths: bool = False) -> list[Path]:
    """Extract a ZIP archive. junk_paths flattens entries into out_dir, like `unzip -j`.
    Returns the list of extracted file paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Extracting {zip_path} -> {out_dir}")
    extracted: list[Path] = []
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            target_name = Path(member.filename).name if junk_paths else member.filename
            target = out_dir / target_name
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            logger.debug(f"Extracted {member.filename} ({member.file_size:,} bytes) -> {target}")
            extracted.append(target)
    logger.info(f"Extracted {len(extracted)} files from {zip_path}")
    return extracted


def remove_files(paths: Iterable[Path]) -> None:
    """Delete intermediate files (e.g. ZIP-extracted CSVs) once they're no longer needed,
    then prune any directories that extracting them created and are now empty (e.g. a
    preserved clean_files/<Type>/ archive layout, when extract_zip wasn't given junk_paths)."""
    paths = list(paths)
    parent_dirs = {path.parent for path in paths}
    for path in paths:
        path.unlink(missing_ok=True)
        logger.debug(f"Removed intermediate file {path}")
    logger.info(f"Removed {len(paths)} intermediate files")

    for directory in sorted(parent_dirs, key=lambda p: len(p.parts), reverse=True):
        while directory.is_dir():
            try:
                directory.rmdir()
            except OSError:
                break  # not empty (e.g. a sibling extraction shares this parent), stop here
            logger.debug(f"Removed now-empty directory {directory}")
            directory = directory.parent


def write_sqlite_table(df: pd.DataFrame, path: Path, table: str, index_col: str = "state") -> None:
    """Write a target dataset table (tsa/fsa) as an indexed SQLite table under processed/,
    alongside (not instead of) the analyst-facing pickle under interim/ - so EstimationService
    can fetch just the rows it needs per request instead of keeping the whole table resident,
    while notebooks/ad-hoc analysis can still load the full table as a plain DataFrame."""
    df = df.copy()
    df[index_col] = df[index_col].astype(str)
    path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(path) as conn:
        df.to_sql(table, conn, index=False, if_exists="replace")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_{index_col} ON {table}({index_col})")
    logger.info(f"Wrote {len(df):,} rows to {path} (table={table})")


def write_json_list(items: Iterable[str], path: Path) -> None:
    """Write a plain list of strings (e.g. topology column names) as JSON under processed/ -
    the format EstimationService reads. Lighter and more inspectable than joblib for what is
    just a list of strings; the joblib copy under interim/ is left as the analyst-facing one."""
    sorted_items = sorted(items)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(sorted_items, f, indent=2)
    logger.info(f"Wrote {len(sorted_items)} entries to {path}")
