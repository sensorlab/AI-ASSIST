import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.api.estimate import StateResponse
from src.config.settings import get_app_settings
from src.domain.estimation.service import _dataset_paths
from src.services.qdrant.config import get_qdrant_config

PROJECT_DIR = Path(__file__).resolve().parents[1]
API_ENDPOINT = "http://localhost:8000/api/v1/estimate/by-generator"
CCT_THRESHOLD = 0.2


def _lf_path() -> Path:
    data_dir = get_app_settings().data_dir
    if not data_dir.is_absolute():
        data_dir = PROJECT_DIR / data_dir
    lf_path, _, _ = _dataset_paths(data_dir, get_qdrant_config().dataset_name)
    return lf_path


@st.cache_data
def load_lf() -> pd.DataFrame:
    return pd.read_pickle(_lf_path())


def _numeric_cols(lf: pd.DataFrame) -> list[str]:
    candidates = [c for c in lf.columns if not c.startswith("oserv_")]
    return list(lf[candidates].select_dtypes(include=["number"]).columns)


def _perturb_state(lf: pd.DataFrame, state: pd.Series, noise_scale: float, noise_seed: int) -> pd.Series:
    if noise_scale == 0.0:
        return state
    cols = _numeric_cols(lf)
    rng = np.random.default_rng(noise_seed)
    col_stds = lf[cols].std(numeric_only=True).to_numpy(dtype=float)
    noise = rng.normal(0, noise_scale * col_stds, len(cols))
    state = state.copy()
    state[cols] = state[cols].to_numpy(dtype=float) + noise
    return state


def _call_api(state: pd.Series) -> StateResponse:
    res = httpx.post(
        API_ENDPOINT,
        json={"variant": "1.0.0", "state": state.to_dict(), "exclude_uids": []},
        timeout=30,
    )
    res.raise_for_status()
    return StateResponse.model_validate_json(res.text)


def _location_type(location: str) -> str:
    return "line" if location.lower().startswith("line") else "bus"


def _build_neighbor_df(out: StateResponse) -> pd.DataFrame:
    rows = []
    for crit_gen, report in out.outputs.items():
        s = report.summary
        for location, cct in s.cct_weighted_per_location.items():
            rows.append(
                {
                    "Crit_gen": crit_gen,
                    "CCT": cct,
                    "Type": _location_type(location),
                    "Location": location,
                    "weight_mass": round(s.location_weight_mass.get(location, float("nan")), 4),
                    "n_neighbors": s.location_counts.get(location, 0),
                    "cct_pred": s.cct_weighted,
                }
            )
    return pd.DataFrame(rows)


def _strip_chart(neigh: pd.DataFrame) -> go.Figure:
    sorted_order = (
        neigh.drop_duplicates("Crit_gen").set_index("Crit_gen")["cct_pred"].sort_values(ascending=False).index.tolist()
    )

    fig = px.strip(
        neigh,
        x="Crit_gen",
        y="CCT",
        color="Type",
        hover_data=["Location", "weight_mass", "n_neighbors", "cct_pred"],
        category_orders={"Crit_gen": sorted_order},
        color_discrete_map={"bus": "#1f77b4", "line": "#2ca02c"},
        template="plotly_white",
        height=560,
    )

    # Unstable zone shading (CCT < threshold)
    fig.add_hrect(
        y0=0,
        y1=CCT_THRESHOLD,
        fillcolor="red",
        opacity=0.07,
        line_width=0,
        layer="below",
    )

    # Threshold line
    fig.add_hline(
        y=CCT_THRESHOLD,
        line_dash="dash",
        line_color="red",
        line_width=1.5,
        annotation_text="stability threshold 0.2 s",
        annotation_position="top left",
        annotation_font_size=14,
    )

    # Predicted CCT diamonds
    pred_df = neigh.drop_duplicates("Crit_gen").set_index("Crit_gen").reindex(sorted_order).reset_index()
    fig.add_scatter(
        x=pred_df["Crit_gen"],
        y=pred_df["cct_pred"],
        mode="markers",
        marker={"symbol": "diamond", "size": 14, "color": "#FF6B00", "line": {"width": 1, "color": "white"}},
        name="CCT predicted",
    )

    fig.update_layout(
        xaxis_title="Critical generator",
        yaxis_title="CCT [s]",
        yaxis={"rangemode": "tozero", "tickfont": {"size": 16}, "title_font": {"size": 16}},
        xaxis={"tickfont": {"size": 16}, "title_font": {"size": 16}},
        legend_title="Type",
        font={"size": 16},
        hoverlabel={"font": {"size": 15}},
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


# ── UI ──────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="CCT Estimator Demo", layout="wide")
st.title("CCT Estimator — live demo")

tab_config, tab_results = st.tabs(["Configuration", "Results"])

lf = load_lf()
state_ids = lf.index.tolist()

with tab_config:
    st.subheader("State selection")
    state_id = st.selectbox("State ID", state_ids, index=st.session_state.get("state_idx", 0))
    st.session_state.state_idx = state_ids.index(state_id)

    col1, col2 = st.columns([2, 1])
    with col1:
        noise_scale = st.slider("Noise scale (relative std)", 0.0, 0.20, 0.03, 0.01)
    with col2:
        if st.button("Pick random state", use_container_width=True):
            st.session_state.state_idx = int(np.random.randint(len(state_ids)))
            st.session_state.noise_seed = 0
            st.rerun()

    st.divider()
    st.subheader("Auto-refresh")
    auto_refresh = st.checkbox("Enable auto-refresh", value=True)
    refresh_interval = st.slider("Refresh interval (s)", 3, 30, 5, step=1, disabled=not auto_refresh)

noise_seed = st.session_state.get("noise_seed", 0)
state = _perturb_state(lf, lf.loc[state_id], noise_scale, noise_seed)

with tab_config:
    st.caption(f"State **{state_id}** · noise_scale={noise_scale} · noise_seed={noise_seed}")

with tab_results:
    try:
        with st.spinner("Querying API…"):
            out = _call_api(state)

        neigh = _build_neighbor_df(out)
        if neigh.empty:
            st.warning("API returned no neighbors for this state.")
        else:
            st.plotly_chart(_strip_chart(neigh), use_container_width=True)

            st.subheader("Prediction summary per generator")
            summary_rows = []
            for crit_gen, report in out.outputs.items():
                s = report.summary
                summary_rows.append(
                    {
                        "Crit_gen": crit_gen,
                        "CCT predicted": round(s.cct_weighted, 4),
                        "n neighbors": s.n,
                        "n_eff": round(s.n_eff, 1),
                        "dist mean": round(s.distances.get("mean", float("nan")), 4),
                    }
                )
            st.dataframe(pd.DataFrame(summary_rows).set_index("Crit_gen"), use_container_width=True)

    except httpx.ConnectError:
        st.error("Cannot reach API at `localhost:8000`. Start the server with `make serve` (or equivalent).")
    except Exception as exc:
        st.exception(exc)

if auto_refresh:
    time.sleep(refresh_interval)
    st.session_state.noise_seed = noise_seed + 1
    st.rerun()
