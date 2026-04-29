# paper2/config.py
# ─────────────────────────────────────────────────────────────────
# Paper 2 Configuration
# "AI-Driven Intervention Portfolio Optimisation for Industrial
#  Emissions Reduction"
# ─────────────────────────────────────────────────────────────────

import os

# ── Paths ──────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_PATH       = os.path.join(BASE_DIR, "data", "risk_df.csv")       # output from Paper 1
ESG_PATH        = os.path.join(BASE_DIR, "data", "esgdata.csv")        # raw emissions series (Paper 1 input)
INTERVENTION_PATH = os.path.join(BASE_DIR, "data", "interventions.csv")
FIGURES_DIR     = os.path.join(BASE_DIR, "figures")
OUTPUTS_DIR     = os.path.join(BASE_DIR, "outputs")

# Ensure output directories exist
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ── Clustering ─────────────────────────────────────────────────────
K_CLUSTERS      = 3          # K for K-means (Silhouette = 0.72)
CLUSTER_SEED    = 42
N_INIT          = 20         # K-means restarts

# ── Optimisation ───────────────────────────────────────────────────
BUDGET_TOTAL    = 200_000    # fixed budget for LP and RL experiments
BUDGET_PER_FAC  = 150_000    # per-facility cap — forces budget to spread across ≥2 facilities
TARGET_YEAR     = 2030
REDUCTION_TARGET = 0.10      # 10% vs 2022 baseline

# ── Uncertainty ────────────────────────────────────────────────────
N_BOOTSTRAP     = 1_000
CONFIDENCE_LEVEL = 0.90

# ── Misc ───────────────────────────────────────────────────────────
RANDOM_SEED     = 42
