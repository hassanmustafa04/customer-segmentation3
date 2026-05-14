#!/usr/bin/env python3
"""
End-to-end customer segmentation pipeline.

Loads (or generates) data, preprocesses features, fits three clustering families,
evaluates them with internal metrics, saves figures to ``outputs/``, and prints
plain-English segment profiles for the strongest model.
"""

from __future__ import annotations

import os

# Headless / sandbox-safe plotting (must run before matplotlib is imported).
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import sys
import warnings

warnings.filterwarnings("ignore", message=r"Could not save figure")
warnings.filterwarnings("ignore", message=r"Could not save Plotly HTML")
from pathlib import Path

import numpy as np
import pandas as pd

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
from src.preprocessor import fit_transform  # noqa: E402
from src.segment_report import label_from_z, pick_best_model, z_vs_global  # noqa: E402
from src.visualizer import (  # noqa: E402
    plot_cluster_profile_heatmap,
    plot_dbscan_eps_silhouette_grid,
    plot_dendrogram,
    plot_elbow_and_silhouette,
    plot_hierarchical_silhouette_curve,
    plot_pca_scatter,
    plot_radar_comparison,
    plot_silhouette_analysis,
)

RANDOM_STATE = 42
OUTPUT_DIR = ROOT / "outputs"


def print_segment_report(raw: pd.DataFrame, labels: np.ndarray, model_name: str) -> None:
    """Console narrative with compact tables per cluster."""
    zdf = z_vs_global(raw, labels)
    print("\n" + "=" * 88)
    print(f"SEGMENT REPORT — {model_name}")
    print("=" * 88)

    for cluster_id in sorted(np.unique(labels)):
        title, bullets = label_from_z(zdf.loc[cluster_id])
        sub = raw.loc[labels == cluster_id]
        share = 100.0 * len(sub) / len(raw)
        header = f"Cluster {cluster_id} — {title} (~{share:.1f}% of customers)"
        print(f"\n{header}")
        print("-" * len(header))
        for b in bullets:
            print(f"  • {b}")
        desc = sub.mean(numeric_only=True).round(2)
        print("  Key averages:")
        print("   ", desc.to_string().replace("\n", "\n    "))

    print("\n" + "=" * 88 + "\n")


def main() -> None:
    np.random.seed(RANDOM_STATE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading customer dataset…")
    df = load_customers(random_state=RANDOM_STATE)
    print(f"Rows: {len(df):,} | Columns: {list(df.columns)}")

    print("Preprocessing (impute → IQR clip → encode → scale → PCA-2D)…")
    pre = fit_transform(df, random_state=RANDOM_STATE)
    X = pre.X_model
    X_pca = pre.X_pca2
    raw = pre.raw_frame

    print("Fitting clustering models…")
    km = run_kmeans_sweep(X, k_min=2, k_max=10, random_state=RANDOM_STATE)
    db = run_dbscan_search(X)
    agg = run_agglomerative(X, n_min=2, n_max=10, random_state=RANDOM_STATE)

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
    summary_path = OUTPUT_DIR / "metrics_summary.csv"

    print("\nFINAL MODEL COMPARISON")
    print(summary.to_string(index=False))

    try:
        summary.to_csv(summary_path, index=False)
        print(f"\nSaved metrics table → {summary_path}")
    except OSError as exc:
        print(f"\nWarning: could not write metrics CSV ({summary_path}): {exc}")

    # Visual artifacts
    print("\nSaving plots to outputs/…")
    plot_elbow_and_silhouette(km, OUTPUT_DIR)
    plot_silhouette_analysis(X, km.labels, OUTPUT_DIR)
    plot_hierarchical_silhouette_curve(agg.n_clusters_tried, agg.silhouettes, agg.best_n_clusters, OUTPUT_DIR)
    plot_dbscan_eps_silhouette_grid(db.grid_scores, OUTPUT_DIR)
    plot_dendrogram(agg.linkage_matrix, OUTPUT_DIR)

    for labels, stub, title in [
        (km.labels, "pca_kmeans", f"PCA clusters — K-Means (k={km.best_k})"),
        (db.labels, "pca_dbscan", "PCA clusters — DBSCAN"),
        (agg.labels, "pca_agglomerative", f"PCA clusters — Agglomerative (n={agg.best_n_clusters})"),
    ]:
        plot_pca_scatter(X_pca, labels, title, OUTPUT_DIR, stub)

    # Heatmaps / radar use cluster labels on cleaned raw frame (same row order as X)
    plot_cluster_profile_heatmap(raw, km.labels, OUTPUT_DIR, "heatmap_profiles_kmeans.png")
    plot_cluster_profile_heatmap(raw, db.labels, OUTPUT_DIR, "heatmap_profiles_dbscan.png")
    plot_cluster_profile_heatmap(raw, agg.labels, OUTPUT_DIR, "heatmap_profiles_agglomerative.png")

    plot_radar_comparison(raw, km.labels, OUTPUT_DIR, "radar_kmeans.png")
    plot_radar_comparison(raw, db.labels, OUTPUT_DIR, "radar_dbscan.png")
    plot_radar_comparison(raw, agg.labels, OUTPUT_DIR, "radar_agglomerative.png")

    best = pick_best_model(summary, n_samples=len(raw))
    label_map = {
        "KMeans (best k)": (km.labels, "K-Means"),
        "DBSCAN (tuned)": (db.labels, "DBSCAN"),
        "Agglomerative (Ward)": (agg.labels, "Agglomerative (Ward)"),
    }
    best_labels, best_pretty = label_map[best]
    db_noise_frac = db_scores.n_noise / max(len(raw), 1)
    if best != "DBSCAN (tuned)" and db_noise_frac > 0.25:
        print(
            f"\nNote: DBSCAN was not selected as best model because noise points "
            f"account for {db_noise_frac:.1%} of customers (threshold 25%)."
        )
    print(f"\nSelected best model by composite internal metrics: {best_pretty}")
    plot_cluster_profile_heatmap(raw, best_labels, OUTPUT_DIR, "heatmap_profiles_best_model.png")
    plot_radar_comparison(raw, best_labels, OUTPUT_DIR, "radar_best_model.png")
    plot_pca_scatter(X_pca, best_labels, f"PCA clusters — BEST: {best_pretty}", OUTPUT_DIR, "pca_best_model")

    print_segment_report(raw, best_labels, best_pretty)
    print("Done.")


if __name__ == "__main__":
    main()
