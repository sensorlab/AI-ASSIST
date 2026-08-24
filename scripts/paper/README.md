# Manuscript analysis scripts

Every table and figure in the Scientific Reports manuscript is produced by a script in this directory. They are analysis code, not part of the service runtime: nothing under `src/` imports them, and the API never runs them.

Each script resolves the repository root as `Path(__file__).resolve().parents[2]`, so it must stay exactly two levels below the root. Inputs are `datasets/<name>/` (prepared by `uv run ai-assist-prepare <dataset>`) and the intermediate `.joblib`/`.parquet` artifacts in `tmp/` written by `scripts/service/*.py`. Outputs are `results/data/*.csv` and `results/figures/*.tex`, both tracked, so the repository holds the full chain from raw dataset to reported number.

Run them from the repository root, at low priority on a shared machine:

```bash
nice -n 19 uv run python scripts/paper/matched_extratrees_comparison.py
nice -n 19 uv run python scripts/paper/make_burstiness_figure.py
```

The per-script docstring records what it computes, which artifacts it needs and which table or figure it feeds. Scripts whose inputs are the ELES operating-state archive cannot be run from a clean clone: ELES classifies those data as sensitive critical-infrastructure information and they are not redistributable (see the manuscript's Data Availability statement). Their outputs are still committed under `results/data/`, in the aggregated and anonymized form the manuscript reports.

The three `make_*_figure.py` scripts emit standalone `pgfplots` figure bodies into `results/figures/`; those bodies are inlined into the manuscript source rather than included by path.
