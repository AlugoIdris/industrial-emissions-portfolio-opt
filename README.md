# Paper 2 — Intervention Portfolio Optimisation Pipeline

**Reproduce all results in 3 steps:**

```bash
git clone https://github.com/yourusername/emissions-paper2
cd emissions-paper2
pip install -r requirements.txt
jupyter notebook 02_paper2_pipeline.ipynb
```

## Inputs (from Paper 1)
| File | Description |
|------|-------------|
| `data/risk_df.csv` | Per-facility risk table (Paper 1 output) |
| `data/esgdata.csv` | Raw emissions + covariates |
| `data/interventions.csv` | Intervention catalogue |

## Outputs
| File | Description |
|------|-------------|
| `outputs/facility_tiers.csv` | K-means tiers (K=3, Sil=0.72) |
| `outputs/lp_portfolio.csv` | LP-optimised intervention portfolio |
| `outputs/table3_portfolio_risk.csv` | Table 3 (paper-ready) |
| `figures/figure6_kmeans_clusters.png` | Figure 6 |
| `figures/figure8_cost_effectiveness.png` | Figure 8 |
| `figures/figure9_abatement_frontier.png` | Figure 9 |

## Repo Structure
```
paper2/
├── 02_paper2_pipeline.py   ← main entry point (convert to .ipynb)
├── config.py               ← all tuneable parameters
├── requirements.txt
├── src/
│   ├── clustering.py       ← K-means, silhouette, elbow scan
│   ├── interventions.py    ← catalogue loader, cost-effectiveness
│   ├── optimisation.py     ← greedy + LP portfolio solver
│   └── visualization_p2.py ← Figures 6, 7, 8, 9
├── data/                   ← (anonymised, not committed)
├── figures/                ← auto-generated
└── outputs/                ← auto-generated CSV tables
```

**Zenodo DOI:** (to be assigned)
