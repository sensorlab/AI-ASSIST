"""Two-dimensional views of a dataset's operating states, PCA beside t-SNE.

The reports argue about where states sit relative to each other - whether a batch forms its own
cluster, whether it lies on the edge of the sampled cloud - and those claims are made with
distances in the full feature space. This renders the same geometry so it can be looked at.

Both panels are shown because they answer different questions and neither is sufficient alone.
PCA is a linear projection, so relative distances and the shape of the cloud survive it, which is
what an edge claim depends on; it simply cannot separate structure that needs more than two
directions. t-SNE resolves local neighbourhoods far better, which is what a cluster claim
depends on, but it optimises neighbour ranks rather than distances: cluster sizes, the gaps
between clusters and any sense of "far" are not meaningful in its output. Read a claim about
clusters from the right panel and a claim about distance from the left.

Topology columns are excluded, matching the analyses in the reports, and features are
standardized so no unit dominates the projection.

Run from the repository root, e.g.:
    DATASET_NAME=interscada/pl STATE_BATCH_PATTERN='^SET_0925' \
        uv run python scripts/evaluation/state_embedding.py
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.config.logging import configure_logging
from src.config.settings import get_app_settings
from src.services.qdrant.config import get_qdrant_config

logger = logging.getLogger(__name__)

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
# Okabe-Ito, chosen for colourblind safety rather than contrast alone.
EXISTING_COLOR: Final[str] = "#0072B2"
BATCH_COLOR: Final[str] = "#D55E00"
SEED: Final[int] = 42


def main() -> None:
    configure_logging()
    config = get_qdrant_config()
    dataset = config.dataset_name
    slug = dataset.replace("/", "-")
    pattern = os.environ.get("STATE_BATCH_PATTERN", "")

    data_dir = get_app_settings().data_dir
    lf = pd.read_pickle(data_dir / dataset / "interim" / "lf.pkl")
    topology = set(json.loads((data_dir / dataset / "processed" / "topology_cols.json").read_text()))
    continuous = [c for c in lf.columns if c not in topology and pd.api.types.is_numeric_dtype(lf[c])]

    values = lf[continuous].astype(float).to_numpy()
    values = (values - values.mean(axis=0)) / np.where(values.std(axis=0) == 0, 1.0, values.std(axis=0))
    ids = np.array([str(i) for i in lf.index])
    batch = np.array([bool(re.search(pattern, i)) for i in ids]) if pattern else np.zeros(len(ids), dtype=bool)
    logger.info(f"{dataset}: {len(ids)} states, {len(continuous)} continuous features, {int(batch.sum())} in the batch")

    pca = PCA(n_components=2, random_state=SEED)
    projections = [("PCA", pca.fit_transform(values), pca.explained_variance_ratio_)]
    # Perplexity has to stay below the sample size; 30 is the usual default and is fine here.
    perplexity = min(30.0, max(5.0, (len(ids) - 1) / 3))
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=SEED)
    projections.append((f"t-SNE (perplexity {perplexity:g})", tsne.fit_transform(values), None))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (title, xy, variance) in zip(axes, projections, strict=True):
        ax.scatter(
            xy[~batch, 0],
            xy[~batch, 1],
            s=14,
            c=EXISTING_COLOR,
            alpha=0.55,
            linewidths=0,
            label=f"existing ({int((~batch).sum())})",
        )
        if batch.any():
            ax.scatter(
                xy[batch, 0],
                xy[batch, 1],
                s=46,
                c=BATCH_COLOR,
                marker="^",
                edgecolors="black",
                linewidths=0.4,
                label=f"batch {pattern} ({int(batch.sum())})",
            )
        subtitle = f" - {variance[0]:.0%} + {variance[1]:.0%} of variance" if variance is not None else ""
        ax.set_title(f"{title}{subtitle}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="best", fontsize=8, frameon=False)
    fig.suptitle(f"{dataset}: operating states in {len(continuous)} standardized continuous features", fontsize=11)
    fig.tight_layout()

    out_dir = PROJECT_DIR / "datasets" / dataset / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"state_embedding_{slug}.png"
    fig.savefig(out_path, dpi=200)
    logger.info(f"Saved {out_path}")


if __name__ == "__main__":
    main()
