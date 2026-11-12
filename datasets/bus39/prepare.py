"""Prepare the bus39 dataset: download the raw ZIP archive from Zenodo, unpack it and run transform.py."""

from pathlib import Path

import click

from scripts._common import REPO_ROOT, extract_zip, fetch_verified, remove_files, run_script
from src.config.logging import configure_logging

RAW_DIR = REPO_ROOT / "datasets/bus39/raw"
OUT_DIR = REPO_ROOT / "datasets/bus39/interim"
# Pinned to the version record, not the concept DOI, which would follow later versions (doi:10.5281/zenodo.24157276).
ZENODO_URL = "https://zenodo.org/records/24157276/files/data.zip?download=1"
ZENODO_MD5 = "c14fb5431ae379ea470334721cb7eaf7"


@click.command()
@click.option(
    "--raw-dir",
    "raw_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=RAW_DIR,
    show_default=True,
    help="Directory holding data.zip, downloaded from Zenodo if absent",
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
    archive = fetch_verified(ZENODO_URL, raw_dir / "data.zip", ZENODO_MD5)
    extracted = extract_zip(archive, out_dir)
    run_script(REPO_ROOT / "datasets/bus39/transform.py", "--in-dir", str(out_dir), "--out-dir", str(out_dir))
    if cleanup:
        remove_files(extracted)


if __name__ == "__main__":
    main()
