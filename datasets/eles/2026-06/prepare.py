"""Prepare the eles/2026-06 dataset: unpack the raw ZIP archive and run transform.py."""

from pathlib import Path

import click

from scripts._common import REPO_ROOT, extract_zip, remove_files, run_script
from src.config.logging import configure_logging

RAW_DIR = REPO_ROOT / "datasets/eles/2026-06/raw"
OUT_DIR = REPO_ROOT / "datasets/eles/2026-06/interim"


@click.command()
@click.option(
    "--raw-dir",
    "raw_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=RAW_DIR,
    show_default=True,
    help="Directory containing the raw Podatki_DSA.zip",
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
    # Preserve the archive's clean_files/<Type>/ layout (don't junk_paths): a stray
    # misplaced clean_files/Dates/LF_main_9.csv would otherwise collide with the real
    # clean_files/LF/LF_main_9.csv once flattened into a single directory.
    extracted = extract_zip(raw_dir / "Podatki_DSA.zip", out_dir)
    run_script(
        REPO_ROOT / "datasets/eles/2026-06/transform.py",
        "--in-dir",
        str(out_dir / "clean_files"),
        "--out-dir",
        str(out_dir),
    )
    if cleanup:
        remove_files(extracted)


if __name__ == "__main__":
    main()
