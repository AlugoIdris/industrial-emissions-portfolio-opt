# paper2/src/visualization_p2.py
"""
All publication figures for Paper 2.
  Figure 6  – K-means cluster scatter (K=3, Silhouette=0.72)
  Figure 7  – Elbow / Silhouette scan (Appendix)
  Figure 8  – Per-tier cost-effectiveness bar chart
  Figure 9  – Portfolio abatement vs budget frontier
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import logging

logger = logging.getLogger(__name__)

TIER_COLORS = {0: "#4C72B0", 1: "#DD8452", 2: "#C44E52"}
TIER_LABELS = {0: "Low Emitters", 1: "Medium Emitters", 2: "High Emitters"}


# ── Figure 6 ──────────────────────────────────────────────────────

def plot_figure6(cluster_result: dict, tier_df, save_dir: str, scan_df=None):
    """
    Two-panel figure:
      Panel 1 – PCA scatter of the 6-feature standardised matrix,
                coloured by rank-based tier labels, with facility annotations.
      Panel 2 – Silhouette score bar chart across K values (from scan_df),
                with a vertical dashed line marking K=3 as the selected elbow.
    """
    X_pca      = cluster_result["X_pca"]
    labels     = cluster_result["labels"]
    facilities = cluster_result["profile"].index.tolist()
    sil        = cluster_result["silhouette_global"]

    n_panels = 2 if scan_df is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]
    ax1 = axes[0]

    for tier in [0, 1, 2]:
        mask = labels == tier
        ax1.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=TIER_COLORS[tier], label=TIER_LABELS[tier],
            s=90, edgecolors="white", linewidths=0.6, zorder=3
        )
        for i, fac in enumerate(facilities):
            if mask[i]:
                ax1.annotate(
                    fac, (X_pca[i, 0], X_pca[i, 1]),
                    textcoords="offset points", xytext=(4, 4),
                    fontsize=7, color="#333333", zorder=4
                )

    ax1.set_xlabel("PC 1", fontsize=11)
    ax1.set_ylabel("PC 2", fontsize=11)
    ax1.set_title(
        f"Figure 6a. Facility Clustering (PCA projection, K=3)\n"
        f"Rank-based percentile tiers  |  Silhouette = {sil:.2f}",
        fontsize=10, pad=10
    )
    ax1.legend(loc="upper right", fontsize=9, framealpha=0.85)
    ax1.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax1.set_axisbelow(True)

    if scan_df is not None:
        ax2 = axes[1]
        ax2.bar(scan_df["K"], scan_df["Silhouette"],
                color="#4C72B0", width=0.6, alpha=0.8)
        ax2.axvline(3, color="#C44E52", linestyle="--", linewidth=1.5,
                    label="K = 3 (selected)")
        ax2.set_xlabel("K (number of clusters)", fontsize=11)
        ax2.set_ylabel("Silhouette Score", fontsize=11)
        ax2.set_title("Figure 6b. Silhouette Score by K\n(elbow at K = 3)",
                      fontsize=10, pad=10)
        ax2.legend(fontsize=9)
        ax2.grid(axis="y", linestyle="--", alpha=0.35)
        ax2.set_xticks(scan_df["K"])

    path = os.path.join(save_dir, "figure6_kmeans_clusters.png")
    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved Figure 6 → %s", path)
    return path


# ── Figure 7 (Appendix) ───────────────────────────────────────────

def plot_elbow_silhouette(scan_df, save_dir: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(scan_df["K"], scan_df["Inertia"], marker="o", color="#4C72B0")
    ax1.set_xlabel("K"); ax1.set_ylabel("Inertia"); ax1.set_title("Elbow Curve")

    ax2.plot(scan_df["K"], scan_df["Silhouette"], marker="s", color="#DD8452")
    ax2.axvline(3, color="grey", linestyle="--", alpha=0.6, label="K = 3 (selected)")
    ax2.set_xlabel("K"); ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Score by K"); ax2.legend(fontsize=9)

    path = os.path.join(save_dir, "figureS1_elbow_silhouette.png")
    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved Figure S1 → %s", path)
    return path


# ── Figure 8 – Per-tier cost effectiveness ────────────────────────

def plot_cost_effectiveness(portfolio_df, save_dir: str):
    tier_order = list(TIER_LABELS.values())
    means = portfolio_df.groupby("TierLabel")["Avg_CostPerTonne"].mean().reindex(tier_order)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(means.index, means.values,
                  color=[TIER_COLORS[i] for i in range(3)], width=0.5)
    ax.bar_label(bars, fmt="€%.0f/t", padding=3, fontsize=9)
    ax.set_ylabel("Average Cost per tonne CO₂ abated (€)")
    ax.set_title("Figure 8. Intervention Cost-Effectiveness by Facility Tier")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    path = os.path.join(save_dir, "figure8_cost_effectiveness.png")
    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved Figure 8 → %s", path)
    return path


# ── Figure 9 – Abatement frontier ────────────────────────────────

def plot_abatement_frontier(frontier_df, save_dir: str):
    """frontier_df: columns [Budget_EUR, Abatement_tCO2]"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(frontier_df["Budget_EUR"] / 1e6,
            frontier_df["Abatement_tCO2"],
            marker="o", color="#4C72B0", linewidth=2)
    ax.fill_between(frontier_df["Budget_EUR"] / 1e6,
                    frontier_df["Abatement_tCO2"],
                    alpha=0.12, color="#4C72B0")
    ax.set_xlabel("Total Budget (M€)")
    ax.set_ylabel("Portfolio Abatement (tCO₂ / month)")
    ax.set_title("Figure 9. Abatement–Budget Efficiency Frontier")
    ax.grid(True, linestyle="--", alpha=0.35)

    path = os.path.join(save_dir, "figure9_abatement_frontier.png")
    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved Figure 9 → %s", path)
    return path
