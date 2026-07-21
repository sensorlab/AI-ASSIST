"""Prepare the interscada/fr dataset: run transform.py against the raw CSVs."""

from pathlib import Path

import click

from scripts._common import REPO_ROOT, run_script
from src.config.logging import configure_logging

RAW_DIR = REPO_ROOT / "datasets/interscada/fr/raw"
OUT_DIR = REPO_ROOT / "datasets/interscada/fr/interim"


@click.command()
@click.option(
    "--raw-dir",
    "raw_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=RAW_DIR,
    show_default=True,
    help="Directory containing the raw interscada/fr CSVs",
)
@click.option(
    "--out-dir",
    "out_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=OUT_DIR,
    show_default=True,
    help="Output directory for interim ML-ready files",
)
def main(raw_dir: Path, out_dir: Path):
    configure_logging()
    run_script(REPO_ROOT / "datasets/interscada/fr/transform.py", "--in-dir", str(raw_dir), "--out-dir", str(out_dir))


if __name__ == "__main__":
    main()
