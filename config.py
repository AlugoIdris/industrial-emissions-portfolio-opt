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
# BUDGET_TOTAL is set to 30% of the lp_min_budget() solution (€5,085,821),
# which is the minimum investment to achieve portfolio P ≥ 0.80 across all
# 30 facilities (first-order Taylor linearisation of norm.cdf).
# 30% = €1,525,746 → rounded to €1,500,000 as the "moderately constrained" scenario.
# BUDGET_PER_FAC is set high (non-binding) so the global budget constraint
# drives allocation rather than a per-facility cap.
BUDGET_TOTAL    = 1_500_000  # 30% of full-compliance cost (€5,085,821)
BUDGET_PER_FAC  = 500_000    # per-facility cap — non-binding; global budget constraint dominates
TARGET_YEAR     = 2030
REDUCTION_TARGET = 0.10      # 10% vs 2022 baseline

# ── Uncertainty ────────────────────────────────────────────────────
N_BOOTSTRAP     = 1_000
CONFIDENCE_LEVEL = 0.90

# ── Misc ───────────────────────────────────────────────────────────
RANDOM_SEED     = 42
