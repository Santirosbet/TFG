# Predictive Analysis of Respiratory Medicine Demand

**TFG — Business Intelligence & Data Analytics**  
Santiago Rosales Betancourt | Supervisor: Otilio Jose Rojas Ulacio | Defensa: Mayo 2026

---

## What this project does

Develops a machine-learning forecasting system for European pharmaceutical demand
(R03 respiratory / R06 antihistamine) that exploits the six-month epidemiological
lead between Southern Hemisphere and Northern Hemisphere influenza seasons.

**Core hypothesis**: Australian flu data (available in July) predicts European
winter demand (January–February) with ~28 weeks advance notice — enough time to
authorise manufacturing batch increases.

**Validated bidirectionally**: AU→EU r = 0.73 (lag +28w) and EU→AU r = 0.70
(lag −28w, p < 0.001, n = 1,531 weeks). Pattern replicated across 7 countries
and 4 continents.

---

## Key results

| Model | MAPE | Notes |
|---|---|---|
| SARIMA | 56.81% | Statistical baseline |
| XGBoost A (lags only) | 46.45% | ML baseline — no epidemiological signal |
| XGBoost B (+ AU flu lag 26-28w) | 44.16% | Core AU signal integrated |
| Prophet | 48.32% | Third baseline |
| LightGBM | ~44% | Ensemble candidate |
| Stacking (XGB + LGB + Ridge) | ~43% | Meta-learner ensemble |
| **Switching Rule** | **35.78%** | **Best overall result** |
| Enriched XGBoost (32 features) | 39.4% avg seasonal | Temp + RSV + Trends + AU + AR |

**Season-by-season backtest (enriched model):**

| Season | MAPE | Total demand error | Peak timing error |
|---|---|---|---|
| 2016–17 | 42.3% | +9.9% | 4 weeks |
| 2017–18 | 37.9% | −18.1% | 5 weeks |
| 2018–19 | 38.1% | −33.5% (severe season) | 4 weeks |
| **Average** | **39.4%** | **−13.9%** | **4.3 weeks** |

> Note: MAPE 35–39% is consistent with the pharmaceutical demand forecasting
> literature (20–40% for single-item, single-location). The most actionable metric
> is **total seasonal demand error (9.9–18.1% in normal years)**, sufficient for
> manufacturing batch decisions.

---

## Dashboard

21-page interactive Streamlit dashboard with dark theme, real-time WHO FluNet data,
and full Spanish/English toggle.

```bash
python -m streamlit run dashboard/app.py
# Opens at http://localhost:8502
```

**Pages include:** Australia Now (live WHO signal → EU projection), Executive Summary,
Order Calculator, Demand Prediction, Lead-Lag Analysis, Hemispheric Validation,
Model Performance, Walk-Forward Validation, Season Backtest, Inventory Simulation,
SHAP Explainability, AMELI France Context, Ensemble & Switching Rule, Robustness
Analysis, and more.

---

## Project structure

```
data_project/
├── config/
│   └── settings.py                     # Centralised config — all paths (no secrets)
├── data/
│   ├── raw/                            # Downloaded source files (NOT committed)
│   ├── processed/                      # Integrated datasets
│   └── external/                       # Google Trends, weather signals
├── src/
│   ├── 01_download_data.py             # WHO FluNet global download
│   ├── 02_clean_flunet.py              # FluNet aggregation by hemisphere
│   ├── 03_clean_european_sales.py      # EU pharma sales cleaning
│   ├── 04_clean_pbs_australia.py       # PBS Australia cleaning
│   ├── 05_integrate_datasets.py        # Lag engineering, feature construction
│   ├── 05b_integrate_v2.py             # Integrate external signals → dataset_v2
│   ├── 06_validate_lead_lag.py         # CCF cross-correlation analysis
│   ├── 07_model_xgboost.py             # XGBoost Model A & B
│   ├── 07b_model_lightgbm.py           # LightGBM + DM test vs XGBoost
│   ├── 07c_model_xgboost_v3.py         # XGBoost Model B enriched
│   ├── 08_download_france_openmedic.py # AMELI France data
│   ├── 09_shap_analysis.py             # SHAP explainability
│   ├── 10_walk_forward_validation.py   # Walk-forward validation (123 folds)
│   ├── 11_sarima_baseline.py           # SARIMA + Ljung-Box test
│   ├── 12_inventory_simulation.py      # (s,Q) policy simulation
│   ├── 13_southern_hemisphere_analysis.py  # 7-country SH validation
│   ├── 14_eda_figures.py               # EDA visualisations
│   ├── 15_r06_model.py                 # R06 model + Diebold-Mariano
│   ├── 16_ensemble_model.py            # Multi-country SH ensemble
│   ├── 16b_stacking_ensemble.py        # Stacking ensemble (Ridge meta-learner)
│   ├── 17_switching_ci.py              # Seasonal switching rule + CI
│   ├── 17b_conformal_prediction.py     # Conformal CI calibration
│   ├── 18_prophet_model.py             # Prophet baseline + DM test
│   ├── 19_google_trends.py             # Google Trends respiratory terms
│   ├── 20_weather_signals.py           # Open-Meteo temp & humidity (7 cities)
│   ├── 21_n02be_model.py               # Paracetamol (N02BE) model
│   ├── 28_robustness_analysis.py       # Robustness analysis
│   ├── 29_enriched_model.py            # Enriched model (32 features)
│   ├── 30_backtest_visualization.py    # Season backtest visualisations
│   ├── build_thesis.py                 # Generates TFG_Final_v4.docx
│   └── build_dummies.py                # Generates TFG for Dummies.docx
├── dashboard/
│   └── app.py                          # 21-page Streamlit dashboard
├── .streamlit/
│   └── config.toml                     # Dark theme configuration
├── output/                             # Generated figures, CSVs, JSONs (NOT committed)
├── .env.example                        # Template for environment variables
├── run_pipeline.py                     # Orchestrates all pipeline steps
└── requirements.txt
```

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/Santirosbet/TFG.git
cd TFG

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables (optional — only needed for Notion sync)
copy .env.example .env
```

Python 3.10+ recommended.

---

## Running the pipeline

```bash
# Full pipeline — all steps in order
python run_pipeline.py

# Run a single script directly
python src/29_enriched_model.py
```

**Step dependency order:**
```
Steps 01-05  → core data download & integration
Step  06     → requires 05 (CCF validation)
Steps 07-18  → requires 05, 06 (models, validation, baselines)
Steps 19-20  → independent (external signals download)
Step  05b    → requires 05, 19, 20 (dataset_v2 integration)
Steps 07b-c  → requires 05b (enriched models)
Steps 28-30  → requires prior model outputs (robustness, enriched, backtest viz)
```

---

## Running the dashboard

```bash
python -m streamlit run dashboard/app.py
# Opens at http://localhost:8502
```

Requires `output/` to contain pipeline outputs (generated by `run_pipeline.py`).

---

## Data sources

| Dataset | Source | Coverage | Cost |
|---|---|---|---|
| WHO FluNet | xmart-api-public.who.int | 140+ countries, 1995–present, weekly | Free |
| EU Pharma Sales | Kaggle (Zdravkovic, 2019) | 1 European pharmacy, 2014–2019, weekly | Free |
| PBS Australia | pbs.gov.au | National prescriptions, monthly | Free |
| Open Medic AMELI | data.gouv.fr | French national market, 2014–2024, annual | Free |
| Google Trends | trends.google.com (pytrends) | Spain/Europe, 2012–2020, weekly | Free |
| Open-Meteo | archive-api.open-meteo.com | 7 European cities, 2012–2020 | Free |

**Total data cost: EUR 0**

---

## Security

- No API keys, tokens, or personal paths are committed to this repository.
- All secrets are loaded from `.env` (see `.env.example`).
- `data/raw/`, `data/processed/`, `data/external/`, and `output/` are in `.gitignore`.

---

## Generating thesis documents

```bash
python src/build_thesis.py    # → output/TFG_Final_v4.docx
python src/build_dummies.py   # → output/TFG for Dummies.docx
```

---

## License

Academic project — Universidad [nombre]. All rights reserved.  
Data sources used under their respective open-data licences (WHO, Kaggle, data.gouv.fr).
