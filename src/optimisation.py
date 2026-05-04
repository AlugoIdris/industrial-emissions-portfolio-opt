# paper2/src/optimisation.py
"""
Portfolio optimisation: maximise total abatement subject to budget constraints.

Strategy 1 — Greedy (fast, interpretable baseline):
  Sort by cost-effectiveness; allocate greedily until budget exhausted.

Strategy 2 — LP (scipy.optimize.linprog, exact):
  Binary selection problem relaxed to continuous for tractability.
  Each intervention is either fully funded (x=1) or not (x=0).
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional
from scipy.optimize import linprog
from scipy.stats import norm as sp_norm

logger = logging.getLogger(__name__)


# ── Greedy ─────────────────────────────────────────────────────────

def greedy_portfolio(candidates: pd.DataFrame,
                     budget: float,
                     per_fac_cap: float) -> pd.DataFrame:
    """
    candidates : output of compute_cost_effectiveness(), one row = one option
    Returns DataFrame of selected interventions with 'Selected' column.
    """
    candidates = candidates.copy().sort_values("CostPerTonne")
    spent       = {}   # facility → cumulative spend
    selected    = []

    for _, row in candidates.iterrows():
        fac  = row["Facility"]
        cost = row["Cost_EUR"]
        spent.setdefault(fac, 0.0)

        if (sum(spent.values()) + cost <= budget) and \
           (spent[fac]             + cost <= per_fac_cap):
            spent[fac] += cost
            selected.append(row.name)

    candidates["Selected"] = candidates.index.isin(selected)
    total_abatement = candidates.loc[candidates["Selected"], "AbatementVolume_tCO2"].sum()
    total_cost      = candidates.loc[candidates["Selected"], "Cost_EUR"].sum()
    logger.info(
        "Greedy portfolio: %d interventions | abatement=%.1f tCO2 | cost=€%.0f",
        candidates["Selected"].sum(), total_abatement, total_cost
    )
    return candidates


# ── LP ─────────────────────────────────────────────────────────────

def lp_portfolio(candidates: pd.DataFrame,
                 budget: float,
                 per_fac_cap: float,
                 risk_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    LP relaxation — maximize portfolio P improvement under a fixed budget.

    Objective: maximise sum_i( weight_f(i) × (AbatementVolume_i / 60) × x_i )

    Weight formulation (scale-invariant):
        z_f     = gap_f / uncertainty_f        (standardised distance to target)
        weight_f = exp(-0.5 × z_f²)            if gap_f > 0
        weight_f = 0                            if gap_f ≤ 0

    Using exp(-½z²) instead of norm.pdf removes the 1/σ scaling factor that
    otherwise makes low-uncertainty facilities dominate the objective regardless
    of gap size.  Facilities closest to the target boundary in sigma units get
    the highest marginal weight (= highest probability bang per tCO2 reduced).

    Constraints:
        1. Global budget:  sum(Cost_i × x_i) ≤ budget
        2. Per-facility budget cap
        3. Physical abatement cap:
               sum_i(Ab_i × x_i) ≤ Prediction_f   for each at-risk facility
           Prevents spending beyond what is physically reducible.

    On-track facilities (gap ≤ 0) are hard-blocked via x bounds (0, 0).
    P_post is clamped to max(pred_f − monthly_ab, 0) in post-processing.
    """
    candidates = candidates.copy().reset_index(drop=True)
    n = len(candidates)
    facilities = candidates["Facility"].unique()

    # ── Per-facility gap and scale-invariant weights ──────────────────
    if risk_df is not None:
        fac_risk = risk_df.set_index("Facility")[
            ["Prediction", "Target_2030", "Uncertainty_sigma", "P_i_percent"]
        ]

        def _weight(r):
            gap = r["Prediction"] - r["Target_2030"]
            if gap <= 0:
                return 0.0
            sigma = r["Uncertainty_sigma"]
            if sigma <= 0:
                return 0.0
            z = gap / sigma
            return float(np.exp(-0.5 * z * z))

        weights = fac_risk.apply(_weight, axis=1)
        gaps    = (fac_risk["Prediction"] - fac_risk["Target_2030"]).clip(lower=0.0)

        candidates["weight_f"] = candidates["Facility"].map(weights).fillna(0.0)
        candidates["gap_f"]    = candidates["Facility"].map(gaps).fillna(0.0)
        candidates["p_pre"]    = candidates["Facility"].map(
            fac_risk["P_i_percent"]
        ).fillna(0.0)
    else:
        candidates["weight_f"] = 1.0
        candidates["gap_f"]    = np.inf
        candidates["p_pre"]    = 0.0

    # Objective: negate because linprog minimises
    c_obj = -(
        candidates["weight_f"] * candidates["AbatementVolume_tCO2"] / 60.0
    ).values

    # ── Constraints ──────────────────────────────────────────────────
    # 1. Global budget
    A_ub = [candidates["Cost_EUR"].values.tolist()]
    b_ub = [budget]

    # 2. Per-facility budget cap
    for fac in facilities:
        fac_mask = (candidates["Facility"] == fac).values.astype(float)
        A_ub.append((fac_mask * candidates["Cost_EUR"].values).tolist())
        b_ub.append(per_fac_cap)

    # 3. Physical abatement cap per at-risk facility:
    #    monthly abatement ≤ Prediction_f  (can't reduce below zero)
    if risk_df is not None:
        for fac in facilities:
            fac_mask = (candidates["Facility"] == fac).values.astype(float)
            ab_row   = (fac_mask * candidates["AbatementVolume_tCO2"].values).tolist()
            gap_f    = float(candidates.loc[candidates["Facility"] == fac, "gap_f"].iloc[0])
            if gap_f <= 0:
                continue  # blocked via bounds anyway
            pred_f = float(fac_risk.loc[fac, "Prediction"])
            A_ub.append(ab_row)
            b_ub.append(pred_f)  # hard physical ceiling

    # On-track facilities: force x=0 via bounds
    bounds = [
        (0.0, 0.0) if candidates.loc[i, "gap_f"] <= 0.0 else (0.0, 1.0)
        for i in range(n)
    ]

    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if res.status != 0:
        logger.warning("LP did not converge: %s", res.message)

    candidates["x_lp"]     = res.x if res.status == 0 else np.nan
    candidates["Selected"]  = candidates["x_lp"] >= 0.5

    # ── Post-intervention risk per facility ───────────────────────────
    if risk_df is not None:
        for fac in facilities:
            fac_mask = candidates["Facility"] == fac
            monthly_abat = (
                candidates.loc[fac_mask, "x_lp"]
                * candidates.loc[fac_mask, "AbatementVolume_tCO2"]
            ).sum()

            pred_f = fac_risk.loc[fac, "Prediction"]
            tgt_f  = fac_risk.loc[fac, "Target_2030"]
            unc_f  = fac_risk.loc[fac, "Uncertainty_sigma"]

            pred_post = max(pred_f - monthly_abat, 0.0)
            p_post    = sp_norm.cdf(tgt_f, loc=pred_post, scale=unc_f)

            candidates.loc[fac_mask, "pred_post"] = pred_post
            candidates.loc[fac_mask, "P_post"]    = p_post

    total_abatement = (candidates["x_lp"] * candidates["AbatementVolume_tCO2"]).sum()
    logger.info(
        "LP portfolio: abatement=%.1f tCO2/month | cost=€%.0f",
        total_abatement, (candidates["x_lp"] * candidates["Cost_EUR"]).sum(),
    )
    return candidates


# ── Min-Budget LP ──────────────────────────────────────────────────

def lp_min_budget(
    candidates: pd.DataFrame,
    risk_df: pd.DataFrame,
    p_target: float = 0.80,
) -> tuple:
    """
    Cost-minimisation LP: find the minimum budget to achieve a portfolio-level
    P(meet 2030 target) ≥ p_target across all N facilities.

    The nonlinear norm.cdf constraint is linearised via a first-order Taylor
    expansion at the pre-intervention operating point:

        P_post_f ≈ P_pre_f + weight_f × monthly_abatement_f

    where weight_f = norm.pdf(Target_f, loc=Pred_f, scale=σ_f).

    LP formulation
    --------------
    Decision variables: x_f_i ∈ [0, 1]  (fractional allocation per candidate)

    Minimise:  sum_{f,i}( Cost_f_i × x_f_i )

    Subject to:
      1. Portfolio:
           sum_{f,i}( weight_f × Ab_f_i × x_f_i ) ≥ N × (p_target - mean_P_pre)
      2. Physical (no negative pred_post per facility):
           sum_i( Ab_f_i × x_f_i ) ≤ Prediction_f   for each f
      3. Bounds: x_f_i ∈ [0, 1]

    Returns
    -------
    optimal_budget : float
        Minimum total cost (EUR) to achieve p_target.
    detail_df : pd.DataFrame
        Row per (facility × intervention) with allocation details.
        Includes all facilities (zero-allocation rows for non-intervened ones).
    portfolio_p_actual : float
        Realised portfolio P using full norm.cdf (vs. the linearised approx).
    """
    candidates = candidates.copy().reset_index(drop=True)
    n = len(candidates)

    fac_risk = risk_df.set_index("Facility")[
        ["Prediction", "Target_2030", "Uncertainty_sigma", "P_i_percent"]
    ]
    N_fac = len(fac_risk)
    mean_p_pre = float(fac_risk["P_i_percent"].mean())

    # Marginal weight = dP/d(monthly_abatement) evaluated at pre-intervention
    weights = fac_risk.apply(
        lambda r: sp_norm.pdf(
            r["Target_2030"], loc=r["Prediction"], scale=r["Uncertainty_sigma"]
        ),
        axis=1,
    )
    candidates["weight_f"] = candidates["Facility"].map(weights)

    # If already above target, return trivial solution
    if mean_p_pre >= p_target:
        logger.info("Portfolio already at P=%.3f ≥ %.2f; no spend needed.",
                    mean_p_pre, p_target)
        detail_rows = []
        for fac in fac_risk.index:
            pred_f  = float(fac_risk.loc[fac, "Prediction"])
            tgt_f   = float(fac_risk.loc[fac, "Target_2030"])
            unc_f   = float(fac_risk.loc[fac, "Uncertainty_sigma"])
            p_pre_f = float(fac_risk.loc[fac, "P_i_percent"])
            detail_rows.append({
                "Facility": fac, "InterventionID": None, "InterventionName": None,
                "Alloc_EUR": 0.0, "Alloc_Abatement_tCO2": 0.0,
                "pred_pre": pred_f, "pred_post": pred_f,
                "p_pre": p_pre_f, "p_post": p_pre_f,
            })
        return 0.0, pd.DataFrame(detail_rows), mean_p_pre

    # ── Compute gaps; zero-out on-track facilities ───────────────────
    gaps = (fac_risk["Prediction"] - fac_risk["Target_2030"]).clip(lower=0.0)
    candidates["gap_f"] = candidates["Facility"].map(gaps).fillna(0.0)

    # ── Objective: minimise total cost ───────────────────────────────
    c_obj = candidates["Cost_EUR"].values.astype(float)

    # ── Constraints ──────────────────────────────────────────────────
    A_ub, b_ub = [], []

    # 1. Portfolio target (≥ → negate for linprog ≤):
    required = N_fac * (p_target - mean_p_pre)
    port_row = -(candidates["weight_f"] * candidates["AbatementVolume_tCO2"]).values
    A_ub.append(port_row.tolist())
    b_ub.append(-required)

    # 2. Physical bound per facility: sum_i(x_fi × Ab_fi) ≤ Prediction_f
    for fac in fac_risk.index:
        fac_mask = (candidates["Facility"] == fac).values.astype(float)
        ab_row   = (fac_mask * candidates["AbatementVolume_tCO2"].values).tolist()
        A_ub.append(ab_row)
        b_ub.append(float(fac_risk.loc[fac, "Prediction"]))

    # On-track facilities (gap ≤ 0): force x = 0 via bounds
    bounds = [
        (0.0, 0.0) if candidates.loc[i, "gap_f"] <= 0.0 else (0.0, 1.0)
        for i in range(n)
    ]
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if res.status != 0:
        logger.warning("Min-budget LP did not converge: %s", res.message)
        return None, None, None

    candidates["x_opt"]          = res.x
    candidates["Alloc_EUR"]       = res.x * candidates["Cost_EUR"]
    candidates["Alloc_Abatement"] = res.x * candidates["AbatementVolume_tCO2"]
    optimal_budget = float(candidates["Alloc_EUR"].sum())

    # ── Actual (nonlinear) P_post per facility ───────────────────────
    detail_rows = []
    for fac in fac_risk.index:
        fac_mask   = candidates["Facility"] == fac
        monthly_ab = float(candidates.loc[fac_mask, "Alloc_Abatement"].sum())
        pred_f     = float(fac_risk.loc[fac, "Prediction"])
        tgt_f      = float(fac_risk.loc[fac, "Target_2030"])
        unc_f      = float(fac_risk.loc[fac, "Uncertainty_sigma"])
        p_pre_f    = float(fac_risk.loc[fac, "P_i_percent"])

        pred_post = max(pred_f - monthly_ab, 0.0)
        p_post    = sp_norm.cdf(tgt_f, loc=pred_post, scale=unc_f)

        sel = candidates.loc[fac_mask & (candidates["x_opt"] > 1e-6)]
        if sel.empty:
            detail_rows.append({
                "Facility": fac, "InterventionID": None, "InterventionName": None,
                "Alloc_EUR": 0.0, "Alloc_Abatement_tCO2": 0.0,
                "pred_pre": pred_f, "pred_post": pred_post,
                "p_pre": p_pre_f, "p_post": p_post,
            })
        else:
            for _, row in sel.iterrows():
                detail_rows.append({
                    "Facility":             fac,
                    "InterventionID":       row["InterventionID"],
                    "InterventionName":     row.get("Name", ""),
                    "Alloc_EUR":            float(row["Alloc_EUR"]),
                    "Alloc_Abatement_tCO2": float(row["Alloc_Abatement"]),
                    "pred_pre":             pred_f,
                    "pred_post":            pred_post,
                    "p_pre":                p_pre_f,
                    "p_post":               p_post,
                })

    detail_df = pd.DataFrame(detail_rows)
    portfolio_p_actual = float(detail_df.groupby("Facility")["p_post"].first().mean())
    n_intervened = int(
        (detail_df.groupby("Facility")["Alloc_EUR"].sum() > 1e-3).sum()
    )
    logger.info(
        "Min-budget LP: optimal=€%.0f | P_actual=%.3f | facilities=%d/%d",
        optimal_budget, portfolio_p_actual, n_intervened, N_fac,
    )
    return optimal_budget, detail_df, portfolio_p_actual


# ── Summary ────────────────────────────────────────────────────────

def portfolio_summary(selected_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate LP-allocated interventions by facility.

    Uses x_lp fractions when available (LP output) so that partial
    allocations are correctly reflected in cost and abatement totals.
    Falls back to the 'Selected' boolean for the greedy portfolio.

    If lp_portfolio was called with risk_df, the output will contain
    P_post (post-intervention probability of meeting target) per facility.
    """
    df = selected_df.copy()
    # Use LP fractions if present, otherwise treat Selected as binary weight
    if "x_lp" in df.columns:
        df["_alloc_cost"] = df["x_lp"] * df["Cost_EUR"]
        df["_alloc_abat"] = df["x_lp"] * df["AbatementVolume_tCO2"]
        active = df[df["x_lp"] > 1e-6]
    else:
        df["_alloc_cost"] = df["Selected"].astype(float) * df["Cost_EUR"]
        df["_alloc_abat"] = df["Selected"].astype(float) * df["AbatementVolume_tCO2"]
        active = df[df["Selected"]]

    agg: dict = dict(
        N_Interventions  = ("InterventionID",  "count"),
        Total_Cost_EUR   = ("_alloc_cost",      "sum"),
        Total_Abatement  = ("_alloc_abat",      "sum"),
        Avg_CostPerTonne = ("CostPerTonne",     "mean"),
    )
    if "P_post" in active.columns:
        agg["P_post"] = ("P_post", "first")
    if "pred_post" in active.columns:
        agg["pred_post"] = ("pred_post", "first")
    summary = active.groupby("Facility").agg(**agg).reset_index()
    return summary
