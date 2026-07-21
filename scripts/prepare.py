"""Prepare a dataset's raw data into ML-ready interim pickles.

Replaces the old per-dataset DVC pipelines (configs/<dataset>/dvc.yaml). Each dataset is
a self-contained directory under `datasets/` holding its raw/interim data alongside its
own `prepare.py` + `transform.py`. Dataset names are discovered by recursively finding
every `prepare.py` under `datasets/`: its path relative to `datasets/`, minus the
filename, is the dataset name, e.g. `datasets/bus39/prepare.py` -> `bus39`,
`datasets/interscada/pl/prepare.py` -> `interscada/pl`. These match the dataset names
used elsewhere (DATASET_NAME/QdrantConfig.dataset_name) - Qdrant collection naming
replaces the slash with a dash on its own, so no back-and-forth translation is needed here.
Adding a new dataset (or a new version of an existing one) means dropping in a new
`datasets/<name>/prepare.py` (or `datasets/<name>/<version>/prepare.py`) - no changes here.

Usage:
    uv run ai-assist-prepare <dataset>
    uv run python scripts/prepare.py <dataset> [--raw-dir DIR] [--out-dir DIR]
"""

import logging
from pathlib import Path

import click

from scripts._common import REPO_ROOT, run_script
from src.config.logging import configure_logging

logger = logging.getLogger(__name__)


def discover_datasets() -> dict[str, Path]:
    datasets_dir = REPO_ROOT / "datasets"
    datasets: dict[str, Path] = {}
    for prepare_script in sorted(datasets_dir.rglob("prepare.py")):
        key = "/".join(prepare_script.relative_to(datasets_dir).parts[:-1])
        datasets[key] = prepare_script
    return datasets


DATASETS = discover_datasets()


@click.command()
@click.argument("dataset", type=click.Choice(sorted(DATASETS)))
@click.option(
    "--raw-dir",
    "raw_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Override the dataset's default raw directory",
)
@click.option(
    "--out-dir",
    "out_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Override the dataset's default output directory",
)
@click.option(
    "--cleanup/--no-cleanup",
    default=None,
    help=(
        "Remove ZIP-extracted intermediate files afterwards (dataset's own default if unset). "
        "Only supported by ZIP-based datasets (currently bus39, eles/2026-01) - passing this "
        "for a dataset without a --cleanup option of its own will error."
    ),
)
def main(dataset: str, raw_dir: Path | None, out_dir: Path | None, cleanup: bool | None):
    configure_logging()
    args = []
    if raw_dir is not None:
        args += ["--raw-dir", str(raw_dir)]
    if out_dir is not None:
        args += ["--out-dir", str(out_dir)]
    if cleanup is not None:
        args += ["--cleanup" if cleanup else "--no-cleanup"]
    logger.info(f"Dispatching '{dataset}' to {DATASETS[dataset]}")
    run_script(DATASETS[dataset], *args)


if __name__ == "__main__":
    main()
