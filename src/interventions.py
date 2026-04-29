# paper2/src/interventions.py
"""
Intervention catalogue loader, tier filtering, and cost-effectiveness
computation for the Paper 2 portfolio optimisation pipeline.

interventions.csv schema (one row per intervention type):
  InterventionID   – unique ID e.g. "EE-01"
  Name             – human-readable label
  Category         – Energy Efficiency | Renewable Energy |
                     Fuel Switching | Process Optimisation |
                     Waste & Circular Economy
  Cost_EUR         – upfront capital cost (€)
  AbatementRate    – fraction of facility baseline monthly emissions
                     reduced (e.g. 0.12 = 12%)
  MinTier          – lowest cluster tier this applies to (0 = Low)
  MaxTier          – highest cluster tier this applies to (2 = High)
  Lifespan_years   – effective life of the measure (for annualisation)
  Source           – bibliographic citation (IEA / EU taxonomy etc.)
  Description      – short plain-English description

Derived columns (added by compute_cost_effectiveness):
  AbatementVolume_tCO2  – AbatementRate × facility_baseline (tCO2/month)
  AnnualisedCost_EUR    – Cost_EUR / Lifespan_years
  CostPerTonne          – AnnualisedCost_EUR / AbatementVolume_tCO2
  CostPerTonne_Lifetime – Cost_EUR / (AbatementVolume_tCO2 × Lifespan_years × 12)
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# ── Required columns in interventions.csv ─────────────────────────
REQUIRED_COLS = [
    "InterventionID", "Name", "Category",
    "Cost_EUR", "AbatementRate",
    "MinTier", "MaxTier", "Lifespan_years",
]

# ── Category colour map (for visualisation) ───────────────────────
CATEGORY_COLORS = {
    "Energy Efficiency":        "#4C72B0",
    "Renewable Energy":         "#55A868",
    "Fuel Switching":           "#DD8452",
    "Process Optimisation":     "#C44E52",
    "Waste & Circular Economy": "#8172B2",
}


# ── Loader ─────────────────────────────────────────────────────────

def load_interventions(path: str) -> pd.DataFrame:
    """
    Load and validate the intervention catalogue CSV.

    Returns a clean DataFrame with correct dtypes.
    Raises ValueError if required columns are missing.
    """
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"interventions.csv is missing required columns: {missing}\n"
            f"Found columns: {df.columns.tolist()}"
        )

    # Enforce dtypes
    df["Cost_EUR"]       = pd.to_numeric(df["Cost_EUR"],       errors="coerce")
    df["AbatementRate"]  = pd.to_numeric(df["AbatementRate"],  errors="coerce")
    df["MinTier"]        = pd.to_numeric(df["MinTier"],        errors="coerce").astype(int)
    df["MaxTier"]        = pd.to_numeric(df["MaxTier"],        errors="coerce").astype(int)
    df["Lifespan_years"] = pd.to_numeric(df["Lifespan_years"], errors="coerce")

    bad = df[df["AbatementRate"].isna() | df["Cost_EUR"].isna()]
    if len(bad):
        logger.warning(
            "Dropping %d rows with missing Cost_EUR or AbatementRate:\n%s",
            len(bad), bad["InterventionID"].tolist()
        )
        df = df.dropna(subset=["AbatementRate", "Cost_EUR"])

    logger.info(
        "Interventions loaded: %d options across %d categories",
        len(df), df["Category"].nunique()
    )
    return df.reset_index(drop=True)


# ── Tier Filter ────────────────────────────────────────────────────

def filter_by_tier(interventions: pd.DataFrame, tier: int) -> pd.DataFrame:
    """
    Return only interventions applicable to a given facility tier.

    An intervention is applicable when:
        MinTier <= tier <= MaxTier
    """
    mask = (interventions["MinTier"] <= tier) & (interventions["MaxTier"] >= tier)
    result = interventions[mask].copy()
    logger.debug(
        "Tier %d: %d / %d interventions applicable",
        tier, len(result), len(interventions)
    )
    return result


# ── Cost-Effectiveness ─────────────────────────────────────────────

def compute_cost_effectiveness(interventions: pd.DataFrame,
                                facility_baseline: float) -> pd.DataFrame:
    """
    Compute abatement volume and cost metrics for a given facility.

    Parameters
    ----------
    interventions     : filtered intervention catalogue (from filter_by_tier)
    facility_baseline : mean monthly emissions (tCO2 / month) for the facility

    Returns
    -------
    DataFrame with added columns:
        AbatementVolume_tCO2   – monthly tCO2 abated
        AnnualisedCost_EUR     – Cost_EUR / Lifespan_years
        CostPerTonne           – annualised cost per monthly tonne abated
        CostPerTonne_Lifetime  – lifetime cost per total tonne abated
    """
    df = interventions.copy()

    df["AbatementVolume_tCO2"] = (
        facility_baseline * df["AbatementRate"]
    ).clip(lower=0)

    df["AnnualisedCost_EUR"] = df["Cost_EUR"] / df["Lifespan_years"]

    # Avoid division by zero
    safe_vol = df["AbatementVolume_tCO2"].replace(0, np.nan)

    df["CostPerTonne"] = df["AnnualisedCost_EUR"] / safe_vol

    df["CostPerTonne_Lifetime"] = (
        df["Cost_EUR"] / (safe_vol * df["Lifespan_years"] * 12)
    )

    return df


# ── Summary helpers ────────────────────────────────────────────────

def catalogue_summary(interventions: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary table grouped by Category showing cost and
    abatement rate ranges.  Useful for Table 2 in the paper.
    """
    return (
        interventions
        .groupby("Category")
        .agg(
            N_Interventions = ("InterventionID", "count"),
            Cost_min_EUR    = ("Cost_EUR",      "min"),
            Cost_max_EUR    = ("Cost_EUR",      "max"),
            Rate_min        = ("AbatementRate", "min"),
            Rate_max        = ("AbatementRate", "max"),
            Avg_Lifespan_yr = ("Lifespan_years","mean"),
        )
        .round(3)
        .sort_values("Cost_min_EUR")
        .reset_index()
    )


def get_intervention(interventions: pd.DataFrame,
                     intervention_id: str) -> pd.Series:
    """Look up a single intervention by ID."""
    row = interventions[interventions["InterventionID"] == intervention_id]
    if row.empty:
        raise KeyError(f"InterventionID '{intervention_id}' not found.")
    return row.iloc[0]
