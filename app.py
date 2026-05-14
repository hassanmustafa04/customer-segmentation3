"""
Streamlit dashboard for interactive customer segmentation exploration.

Run from the project root::

    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from streamlit_option_menu import option_menu

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clustering import (  # noqa: E402
    run_agglomerative,
    run_dbscan_search,
    run_kmeans_sweep,
)
from src.data_loader import load_customers  # noqa: E402
from src.evaluator import evaluate_clustering, scores_to_row  # noqa: E402
from src.preprocessor import NUMERIC_FEATURES, fit_transform  # noqa: E402
from src.segment_report import label_from_z, z_vs_global  # noqa: E402

RANDOM_STATE = 42

# Consistent Plotly discrete colors (extended for many clusters)
_BASE_COLORS = (
    px.colors.qualitative.Bold
    + px.colors.qualitative.Dark24
    + px.colors.qualitative.Set2
)


def _hex_to_rgba_fill(hex_color: str, alpha: float = 0.22) -> str:
    try:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"
    except (ValueError, IndexError):
        return "rgba(31, 119, 180, 0.22)"


def _color_to_rgba_fill(color: str, alpha: float = 0.22) -> str:
    c = color.strip()
    if c.startswith("#"):
        return _hex_to_rgba_fill(c, alpha)
    if c.startswith("rgb") and "(" in c:
        inner = c[c.find("(") + 1 : c.find(")")]
        parts = [p.strip() for p in inner.split(",")]
        if len(parts) >= 3:
            return f"rgba({parts[0]},{parts[1]},{parts[2]},{alpha})"
    return "rgba(31, 119, 180, 0.22)"


def _cluster_color_map(labels: np.ndarray) -> dict[str, str]:
    uniq = sorted(np.unique(labels))
    return {str(u): _BASE_COLORS[i % len(_BASE_COLORS)] for i, u in enumerate(uniq)}


def _noise_adjusted_best_silhouette(summary: pd.DataFrame, n_samples: int) -> float:
    work = summary.copy()
    noise_frac = work["n_noise"] / max(n_samples, 1)
    eff = work["silhouette"].where(noise_frac <= 0.25)
    if eff.notna().any():
        return float(eff.max())
    return float(work["silhouette"].max())


def _silhouette_for_labels(X: np.ndarray, labels: np.ndarray) -> float | None:
    mask = labels >= 0
    if mask.sum() < 2 or len(np.unique(labels[mask])) < 2:
        return None
    try:
        return float(silhouette_score(X[mask], labels[mask]))
    except ValueError:
        return None


@st.cache_data(show_spinner="Loading data and fitting clustering models…")
def run_segmentation_pipeline(random_state: int = RANDOM_STATE) -> dict:
    """Load data, preprocess, and fit K-Means sweep, DBSCAN, and hierarchical clustering."""
    df = load_customers(random_state=random_state)
    pre = fit_transform(df, random_state=random_state)
    X = np.asarray(pre.X_model, dtype=np.float64)
    X_pca = np.asarray(pre.X_pca2, dtype=np.float64)
    raw = pre.raw_frame.copy()

    km = run_kmeans_sweep(X, k_min=2, k_max=10, random_state=random_state)
    db = run_dbscan_search(X)
    agg = run_agglomerative(X, n_min=2, n_max=10, random_state=random_state)

    km_scores = evaluate_clustering(X, km.labels, inertia=km.final_inertia)
    db_scores = evaluate_clustering(X, db.labels, inertia=None)
    agg_scores = evaluate_clustering(X, agg.labels, inertia=None)

    summary = pd.DataFrame(
        [
            scores_to_row("KMeans (best k)", km_scores),
            scores_to_row("DBSCAN (tuned)", db_scores),
            scores_to_row("Agglomerative (Ward)", agg_scores),
        ]
    )

    return {
        "raw": raw,
        "X": X,
        "X_pca": X_pca,
        "km_sweep": km,
        "db": db,
        "agg": agg,
        "summary": summary,
        "n_samples": len(raw),
        "km_best_k": int(km.best_k),
    }


@st.cache_data(show_spinner="Fitting K-Means…")
def kmeans_labels_at_k(X: np.ndarray, k: int, random_state: int = RANDOM_STATE) -> np.ndarray:
    model = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    return model.fit_predict(X)


def _active_labels(
    pipe: dict,
    algorithm: str,
    kmeans_k: int,
) -> tuple[np.ndarray, str]:
    if algorithm == "K-Means":
        return kmeans_labels_at_k(pipe["X"], kmeans_k), f"K-Means (k={kmeans_k})"
    if algorithm == "DBSCAN":
        return pipe["db"].labels, "DBSCAN"
    return pipe["agg"].labels, f"Hierarchical (n={pipe['agg'].best_n_clusters})"


def _page_overview(
    pipe: dict,
    labels: np.ndarray,
    algo_label: str,
    scatter_x: str,
    scatter_y: str,
    use_pca: bool,
) -> None:
    raw: pd.DataFrame = pipe["raw"]
    X: np.ndarray = pipe["X"]
    X_pca: np.ndarray = pipe["X_pca"]
    summary: pd.DataFrame = pipe["summary"]
    n = pipe["n_samples"]

    mask_inlier = labels >= 0
    n_clusters = int(len(np.unique(labels[mask_inlier]))) if mask_inlier.any() else 0
    best_sil = _noise_adjusted_best_silhouette(summary, n)
    cur_sil = _silhouette_for_labels(X, labels)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total customers", f"{n:,}")
    with c2:
        st.metric("Clusters (excl. noise)", f"{n_clusters}")
    with c3:
        st.metric(
            "Best silhouette (noise-adjusted)",
            f"{best_sil:.3f}" if pd.notna(best_sil) else "—",
        )
    with c4:
        st.metric(f"Silhouette ({algo_label})", f"{cur_sil:.3f}" if cur_sil is not None else "—")

    st.markdown("---")
    cmap = _cluster_color_map(labels)
    plot_df = pd.DataFrame(
        {
            "x": X_pca[:, 0] if use_pca else raw[scatter_x].astype(float),
            "y": X_pca[:, 1] if use_pca else raw[scatter_y].astype(float),
            "cluster": pd.Series(labels).astype(str),
        }
    )
    xlab = "PC1" if use_pca else scatter_x
    ylab = "PC2" if use_pca else scatter_y

    fig_scatter = px.scatter(
        plot_df,
        x="x",
        y="y",
        color="cluster",
        color_discrete_map=cmap,
        title=f"2D view — {algo_label}",
        labels={"x": xlab, "y": ylab},
        height=520,
        opacity=0.85,
    )
    fig_scatter.update_traces(marker=dict(size=8, line=dict(width=0)))
    fig_scatter.update_layout(template="plotly_white", legend_title_text="Cluster")
    st.plotly_chart(fig_scatter, use_container_width=True)

    vc = pd.Series(labels).astype(str).value_counts().reset_index()
    vc.columns = ["cluster", "count"]
    fig_donut = px.pie(
        vc,
        values="count",
        names="cluster",
        title="Cluster sizes",
        color="cluster",
        color_discrete_map=cmap,
        hole=0.45,
    )
    fig_donut.update_traces(textposition="inside", textinfo="percent+label")
    fig_donut.update_layout(template="plotly_white", height=420, showlegend=True)
    st.plotly_chart(fig_donut, use_container_width=True)


def _page_profiles(pipe: dict, labels: np.ndarray, algo_label: str) -> None:
    raw: pd.DataFrame = pipe["raw"]
    cmap = _cluster_color_map(labels)
    zdf = z_vs_global(raw, labels)
    clusters = sorted(np.unique(labels))

    cid = st.selectbox(
        "Select cluster",
        clusters,
        format_func=lambda x: "Noise (unclustered)" if int(x) == -1 else f"Cluster {x}",
    )
    title, bullets = label_from_z(zdf.loc[int(cid)])

    with st.container():
        st.subheader(f"Cluster {cid} — {title}")
        for b in bullets:
            st.markdown(f"- {b}")

    g_mean = raw[NUMERIC_FEATURES].mean()
    c_mean = raw.loc[labels == cid, NUMERIC_FEATURES].mean()
    theta = NUMERIC_FEATURES
    r_pop = g_mean.reindex(theta).fillna(0).tolist()
    r_clu = c_mean.reindex(theta).fillna(0).tolist()

    fig_radar = go.Figure()
    fig_radar.add_trace(
        go.Scatterpolar(
            r=r_pop + [r_pop[0]],
            theta=theta + [theta[0]],
            fill="toself",
            name="Population avg",
            line_color="#888888",
            fillcolor=_color_to_rgba_fill("#888888"),
        )
    )
    fig_radar.add_trace(
        go.Scatterpolar(
            r=r_clu + [r_clu[0]],
            theta=theta + [theta[0]],
            fill="toself",
            name=f"Cluster {cid}",
            line_color=cmap[str(cid)],
            fillcolor=_color_to_rgba_fill(cmap[str(cid)]),
        )
    )
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        title=f"Radar: cluster {cid} vs population ({algo_label})",
        template="plotly_white",
        height=520,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    comp = pd.DataFrame({"population": g_mean, "cluster": c_mean}).reset_index().rename(columns={"index": "feature"})
    comp_long = comp.melt(id_vars="feature", var_name="series", value_name="value")
    fig_bar = px.bar(
        comp_long,
        x="feature",
        y="value",
        color="series",
        barmode="group",
        title="Feature averages: cluster vs population",
        color_discrete_map={"population": "#888888", "cluster": cmap[str(cid)]},
        height=420,
    )
    fig_bar.update_layout(template="plotly_white", xaxis_tickangle=-35)
    st.plotly_chart(fig_bar, use_container_width=True)


def _page_compare(pipe: dict, labels: np.ndarray, algo_label: str) -> None:
    raw: pd.DataFrame = pipe["raw"]
    summary: pd.DataFrame = pipe["summary"]

    df_h = raw[NUMERIC_FEATURES].copy()
    df_h["cluster"] = labels
    profile = df_h.groupby("cluster")[NUMERIC_FEATURES].mean()
    fig_hm = px.imshow(
        profile,
        labels=dict(x="Feature", y="Cluster", color="Mean (raw units)"),
        title=f"Cluster mean features — {algo_label}",
        color_continuous_scale="Blues",
        aspect="auto",
        height=max(360, 80 * len(profile.index)),
    )
    fig_hm.update_layout(template="plotly_white")
    st.plotly_chart(fig_hm, use_container_width=True)

    st.subheader("Metrics by algorithm")
    disp = summary.copy()
    st.dataframe(disp, use_container_width=True, hide_index=True)


def _page_explorer(pipe: dict, labels: np.ndarray, algo_label: str) -> None:
    raw: pd.DataFrame = pipe["raw"]
    cmap = _cluster_color_map(labels)

    explorer = raw.copy()
    explorer.insert(0, "customer_id", np.arange(len(explorer)))
    explorer["cluster"] = labels

    clist = sorted(np.unique(labels))
    pick = st.multiselect("Clusters", clist, default=clist)
    age_min, age_max = int(explorer["age"].min()), int(explorer["age"].max())
    inc_min, inc_max = float(explorer["income"].min()), float(explorer["income"].max())
    a_lo, a_hi = st.slider("Age range", age_min, age_max, (age_min, age_max))
    i_lo, i_hi = st.slider("Income range", inc_min, inc_max, (inc_min, inc_max), format="%.0f")

    filt = explorer["cluster"].isin(pick) & explorer["age"].between(a_lo, a_hi) & explorer["income"].between(i_lo, i_hi)
    out = explorer.loc[filt].sort_values(["cluster", "customer_id"])

    st.caption(f"{len(out):,} rows after filters · {algo_label}")
    st.dataframe(out, use_container_width=True, height=420)

    csv_bytes = out.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered segment (CSV)",
        data=csv_bytes,
        file_name="segment_export.csv",
        mime="text/csv",
    )


def main() -> None:
    st.set_page_config(page_title="Customer Segmentation", layout="wide", initial_sidebar_state="expanded")

    st.title("Customer segmentation dashboard")
    st.caption("Interactive views over the same pipeline used in `main.py` (cached on first load).")

    pipe = run_segmentation_pipeline(RANDOM_STATE)

    with st.sidebar:
        st.header("Controls")
        algorithm = st.radio(
            "Algorithm",
            ["K-Means", "DBSCAN", "Hierarchical"],
            horizontal=False,
        )
        k_default = int(pipe["km_best_k"])
        kmeans_k = st.slider(
            "Number of clusters (K-Means)",
            min_value=2,
            max_value=10,
            value=k_default,
            disabled=(algorithm != "K-Means"),
            help="Only applies when K-Means is selected. Default matches silhouette-optimal k from sweep.",
        )

        st.subheader("Scatter axes")
        use_pca = st.toggle("Use PCA (PC1 vs PC2)", value=True)
        feat_opts = ["PC1", "PC2"] + NUMERIC_FEATURES
        scatter_x = st.selectbox("Horizontal axis", feat_opts, index=0)
        scatter_y = st.selectbox("Vertical axis", feat_opts, index=1)

    # Resolve axis names for non-PCA mode (ignore PC picks when using raw)
    use_pca_effective = use_pca
    sx, sy = scatter_x, scatter_y
    if not use_pca_effective:
        if scatter_x in ("PC1", "PC2"):
            sx = NUMERIC_FEATURES[0]
        if scatter_y in ("PC1", "PC2"):
            sy = NUMERIC_FEATURES[1] if NUMERIC_FEATURES[1] != sx else NUMERIC_FEATURES[2]
        if sx == sy:
            sy = NUMERIC_FEATURES[1] if NUMERIC_FEATURES[0] == sx else NUMERIC_FEATURES[0]

    labels, algo_label = _active_labels(pipe, algorithm, kmeans_k)

    selected = option_menu(
        None,
        ["Overview", "Cluster Profiles", "Compare Clusters", "Individual Explorer"],
        icons=["house", "clipboard-data", "layout-text-window-reverse", "search"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "nav-link": {"font-size": "15px", "text-align": "left", "margin": "0px"},
            "nav-link-selected": {"background-color": "#1f77b4"},
        },
    )

    if selected == "Overview":
        _page_overview(pipe, labels, algo_label, sx, sy, use_pca_effective)
    elif selected == "Cluster Profiles":
        _page_profiles(pipe, labels, algo_label)
    elif selected == "Compare Clusters":
        _page_compare(pipe, labels, algo_label)
    else:
        _page_explorer(pipe, labels, algo_label)


if __name__ == "__main__":
    main()
