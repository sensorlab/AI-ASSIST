"""Shared helpers for the scripts/*/prepare*.py dataset-preparation scripts."""

import json
import logging
import shutil
import sqlite3
import subprocess
import sys
import zipfile
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


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
