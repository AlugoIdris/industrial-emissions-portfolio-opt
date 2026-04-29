# paper2/src/clustering.py
"""
Two-stage facility clustering for Paper 2.

Stage 1 – K-Means silhouette sweep (data-driven K selection)
  Six features: mean_emissions, std_emissions, slope, mean_production,
  mean_energy, mean_renewable.  Standardised with StandardScaler.
  Sweep K=2..8 to confirm elbow at K=3.

Stage 2 – Rank-based label assignment (deterministic percentile cuts)
  Sort facilities by mean_emissions ascending, then assign:
    bottom 50 % → 0  Low Emitters
    next   40 % → 1  Medium Emitters
    top    10 % → 2  High Emitters
  Guarantees reproducible group sizes regardless of random seed.
"""

import numpy as np
import pandas as pd
import logging
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples

logger = logging.getLogger(__name__)

TIER_LABELS = {0: "Low Emitters", 1: "Medium Emitters", 2: "High Emitters"}

_FEATURES = ["mean_emissions", "std_emissions", "slope",
             "mean_production", "mean_energy", "mean_renewable"]

# Percentile upper boundaries for rank-based cuts
_CUT_LOW = 0.50   # bottom 50 % → Low
_CUT_MED = 0.90   # next 40 % → Medium; remainder → High


def build_facility_profile(df: pd.DataFrame,
                           target_col: str = "Emissions_tCO2") -> pd.DataFrame:
    """
    Compute 6 per-facility features from the raw monthly time series:
      1. mean_emissions  – mean monthly Emissions_tCO2
      2. std_emissions   – std of monthly emissions
      3. slope           – OLS linear trend (tCO2 / month)
      4. mean_production – mean monthly Production
      5. mean_energy     – mean monthly Energy_MWh
      6. mean_renewable  – mean monthly Renewable_percent

    Returns a DataFrame indexed by Facility.
    """
    records = []
    for fac, grp in df.groupby("Facility"):
        grp = grp.sort_values("Date").reset_index(drop=True)
        em  = grp[target_col].values
        t   = np.arange(len(em), dtype=float)
        slope = float(np.polyfit(t, em, 1)[0]) if len(em) > 1 else 0.0
        records.append({
            "Facility":        fac,
            "mean_emissions":  em.mean(),
            "std_emissions":   em.std(ddof=1) if len(em) > 1 else 0.0,
            "slope":           slope,
            "mean_production": grp["Production"].mean(),
            "mean_energy":     grp["Energy_MWh"].mean(),
            "mean_renewable":  grp["Renewable_percent"].mean(),
        })
    profile = pd.DataFrame(records).set_index("Facility").dropna()
    logger.info("Facility profiles built: %d facilities × %d features",
                len(profile), profile.shape[1])
    return profile


def run_kmeans(profile: pd.DataFrame, k: int, seed: int, n_init: int) -> dict:
    """
    Stage 1: standardise the 6-feature matrix, fit K-Means, compute silhouette.
    Also projects to 2 PCA components for visualisation.

    Note: cluster_result["labels"] holds k-means labels here; assign_tiers()
    overwrites them with deterministic rank-based labels.
    """
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(profile[_FEATURES])

    km     = KMeans(n_clusters=k, random_state=seed, n_init=n_init)
    labels = km.fit_predict(X_scaled)

    sil_global  = silhouette_score(X_scaled, labels)
    sil_samples = silhouette_samples(X_scaled, labels)

    pca   = PCA(n_components=2, random_state=seed)
    X_pca = pca.fit_transform(X_scaled)

    result = {
        "profile":           profile,
        "features":          _FEATURES,
        "X_scaled":          X_scaled,
        "X_pca":             X_pca,
        "labels":            labels,
        "centroids_scaled":  km.cluster_centers_,
        "scaler":            scaler,
        "pca":               pca,
        "silhouette_global": sil_global,
        "sil_samples":       sil_samples,
        "km":                km,
    }
    logger.info("K-means K=%d  Silhouette=%.4f", k, sil_global)
    return result


def assign_tiers(cluster_result: dict) -> pd.DataFrame:
    """
    Stage 2: deterministic rank-based label assignment.

    Sort facilities by mean_emissions ascending, then apply fixed cuts:
      bottom 50 % → 0  Low Emitters
      next   40 % → 1  Medium Emitters
      top    10 % → 2  High Emitters

    Overwrites cluster_result["labels"] and cluster_result["sil_samples"]
    so that downstream visualisation uses the rank-based labels.
    """
    profile = cluster_result["profile"]
    n       = len(profile)
    n_low   = int(n * _CUT_LOW)
    n_med   = int(n * (_CUT_MED - _CUT_LOW))

    sorted_facs = profile["mean_emissions"].sort_values().index.tolist()
    tier_map = {}
    for i, fac in enumerate(sorted_facs):
        if i < n_low:
            tier_map[fac] = 0
        elif i < n_low + n_med:
            tier_map[fac] = 1
        else:
            tier_map[fac] = 2

    labels_rank = np.array([tier_map[f] for f in profile.index])
    cluster_result["labels"]      = labels_rank
    cluster_result["sil_samples"] = silhouette_samples(
        cluster_result["X_scaled"], labels_rank)

    df_out = profile.copy()
    df_out["Tier"]         = labels_rank
    df_out["TierLabel"]    = df_out["Tier"].map(TIER_LABELS)
    df_out["Silhouette_i"] = cluster_result["sil_samples"]
    df_out = df_out.sort_values(["Tier", "mean_emissions"])

    counts = df_out["TierLabel"].value_counts()
    logger.info("Tier distribution (rank-based):\n%s", counts.to_string())
    return df_out


def elbow_silhouette_scan(profile: pd.DataFrame, k_range=range(2, 8),
                          seed: int = 42, n_init: int = 10) -> pd.DataFrame:
    """
    Scan K values on the 6-feature standardised matrix.
    Returns DataFrame [K, Inertia, Silhouette].
    """
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(profile[_FEATURES])
    rows = []
    for k in k_range:
        km     = KMeans(n_clusters=k, random_state=seed, n_init=n_init)
        labels = km.fit_predict(X_scaled)
        sil    = silhouette_score(X_scaled, labels) if k > 1 else np.nan
        rows.append({"K": k, "Inertia": km.inertia_, "Silhouette": sil})
    return pd.DataFrame(rows)
