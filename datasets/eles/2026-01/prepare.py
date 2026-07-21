"""Prepare the eles dataset: unpack the raw ZIP archive and run transform.py."""

from pathlib import Path

import click

from scripts._common import REPO_ROOT, extract_zip, remove_files, run_script
from src.config.logging import configure_logging

RAW_DIR = REPO_ROOT / "datasets/eles/2026-01/raw"
OUT_DIR = REPO_ROOT / "datasets/eles/2026-01/interim"


@click.command()
@click.option(
    "--raw-dir",
    "raw_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=RAW_DIR,
    show_default=True,
    help="Directory containing the raw eles ZIP, ignorelist.txt and powerfactory_dictionary.xlsx",
)
@click.option(
    "--out-dir",
    "out_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=OUT_DIR,
    show_default=True,
    help="Output directory for interim ML-ready files",
)
@click.option(
    "--cleanup/--no-cleanup",
    default=True,
    show_default=True,
    help="Remove the ZIP-extracted intermediate CSVs once the pickles are built",
)
def main(raw_dir: Path, out_dir: Path, cleanup: bool):
    configure_logging()
    extracted = extract_zip(raw_dir / "SLOJun2024_Jan2025_only_lne_1h.zip", out_dir, junk_paths=True)
    run_script(
        REPO_ROOT / "datasets/eles/2026-01/transform.py",
        "--in-dir",
        str(out_dir),
        "--ignore-list",
        str(raw_dir / "ignorelist.txt"),
        "--dictionary",
        str(raw_dir / "powerfactory_dictionary.xlsx"),
        "--out-dir",
        str(out_dir),
    )
    if cleanup:
        remove_files(extracted)


if __name__ == "__main__":
    main()
