# Predictive Analysis of Respiratory Medicine Inventory Demand

**TFG — Business Intelligence & Data Analytics**  
Santiago Rosales Betancourt | Supervisor: Otilio Jose Rojas Ulacio

---

## What this project does

This project develops a machine-learning forecasting system for European pharmaceutical demand (R03 respiratory / R06 antihistamine products) that exploits the six-month epidemiological lead between Southern Hemisphere and Northern Hemisphere influenza seasons.

**Core hypothesis**: Australian flu data (available in July) predicts European winter demand (January-February) with ~28 weeks advance notice — enough time to authorise manufacturing batch increases.

**Key results**:
- Cross-correlation r = 0.70 (p < 0.001) between Australian and European flu activity at lag 28 weeks
- XGBoost Model B (lags + WHO FluNet): MAPE = 44.16% on 60-week holdout
- XGBoost vs SARIMA baseline: DM statistic = 6.23, p < 0.001
- Inventory simulation: XGBoost saves EUR 273 over 60 weeks vs naive policy
- Framework validated across 7 Southern Hemisphere countries (Australia, NZ, Chile, Argentina, Brazil, South Africa, Uruguay)

---

## Project structure

```
data_project/
├── data/
│   ├── raw/                   # downloaded source files (not committed)
│   └── processed/             # cleaned, integrated datasets
├── src/
│   ├── 01_download_data.py    # WHO FluNet global download
│   ├── 02_clean_flunet.py     # FluNet aggregation by hemisphere
│   ├── 03_clean_european_sales.py
│   ├── 04_clean_pbs_australia.py
│   ├── 05_integrate_datasets.py   # lag engineering, feature construction
│   ├── 06_validate_lead_lag.py    # cross-correlation analysis (CCF)
│   ├── 07_model_xgboost.py        # XGBoost Model A & B
│   ├── 08_download_france_openmedic.py
│   ├── 09_shap_analysis.py
│   ├── 10_walk_forward_validation.py  # 48-fold expanding window CV
│   ├── 11_sarima_baseline.py          # SARIMA + Ljung-Box test
│   ├── 12_inventory_simulation.py     # (s,Q) policy simulation
│   ├── 13_southern_hemisphere_analysis.py  # 7-country SH validation
│   ├── 14_eda_figures.py              # EDA visualisations
│   ├── 15_r06_model.py                # R06 model + Diebold-Mariano tests
│   └── build_thesis.py                # generates TFG_Final_v2.docx
├── dashboard/
│   └── app.py                 # 10-page Streamlit dashboard
├── output/                    # all generated figures, CSVs, JSONs, .docx
├── run_pipeline.py            # orchestrates all steps in order
└── requirements.txt
```

---

## Setup

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

Python 3.10+ recommended.

---

## Running the pipeline

### Full pipeline (all steps)
```bash
python run_pipeline.py
```

### Resume from a specific step
```bash
python run_pipeline.py --from 9
```

### Run a single step
```bash
python run_pipeline.py --only 11
```

### Run a step directly
```bash
python src/07_model_xgboost.py
```

**Step dependencies** — steps must be run in order (each step reads the previous step's outputs):

| Steps | Dependency |
|-------|-----------|
| 1-4   | Independent (raw data download/cleaning) |
| 5     | Requires 1-4 |
| 6     | Requires 5 |
| 7     | Requires 5, 6 |
| 8     | Independent |
| 9     | Requires 7 |
| 10    | Requires 7 |
| 11    | Requires 7, 10 |
| 12    | Requires 7, 11 |
| 13    | Requires 5, 7 |
| 14    | Requires 5, 8 |
| 15    | Requires 7, 11, 10 |

---

## Running the dashboard

```bash
streamlit run dashboard/app.py
```

Opens at http://localhost:8501. Requires `output/` to contain all pipeline outputs.

---

## Generating the thesis document

```bash
python src/build_thesis.py
```

Outputs `output/TFG_Final_v2.docx`. Figures from `output/*.png` are automatically embedded.

---

## Data sources

| Dataset | Source | Coverage |
|---------|--------|----------|
| WHO FluNet | xmart-api-public.who.int | 140+ countries, 1995-present, weekly |
| EU Pharma Sales | Kaggle (Zdravkovic, 2019) | 1 European pharmacy, 2014-2019, weekly |
| PBS Australia | pbs.gov.au | National prescriptions, 2020-2024, monthly |
| Open Medic AMELI | data.gouv.fr | National French market, 2014-2024, annual |

---

## Key outputs

| File | Description |
|------|-------------|
| `output/xgboost_best_model.json` | Serialised XGBoost model (production-ready) |
| `output/test_predictions.csv` | 60-week holdout predictions vs actual |
| `output/wfv_predictions.csv` | 192 walk-forward OOS predictions |
| `output/sarima_predictions.csv` | SARIMA 60-week forecast with 80% CI |
| `output/inventory_simulation.csv` | Week-by-week inventory state (4 scenarios) |
| `output/sh_model_comparison.csv` | 7-country Southern Hemisphere results |
| `output/dm_test_results.json` | Diebold-Mariano significance test results |
| `output/TFG_Final_v2.docx` | Full thesis document with embedded figures |
