"""
Full pipeline runner — executes all 17 analysis steps in order.

Usage:
    python run_pipeline.py            # run all steps
    python run_pipeline.py --from 6   # resume from step 6
    python run_pipeline.py --only 11  # run a single step

Each step is a standalone script in src/. Steps are numbered and must run
in order because each step's output is consumed by later steps.
"""

import subprocess
import sys
import os
import time

STEPS = [
    # ── Pipeline original (18 pasos) ──────────────────────────────────────────
    (1,  "src/01_download_data.py",              "Download WHO FluNet global data"),
    (2,  "src/02_clean_flunet.py",               "Clean and aggregate FluNet by hemisphere"),
    (3,  "src/03_clean_european_sales.py",        "Clean European pharmacy sales data"),
    (4,  "src/04_clean_pbs_australia.py",         "Clean Australian PBS prescription data"),
    (5,  "src/05_integrate_datasets.py",          "Integrate all datasets + lag engineering"),
    (6,  "src/06_validate_lead_lag.py",           "Cross-correlation analysis (CCF)"),
    (7,  "src/07_model_xgboost.py",              "Train XGBoost Model A & B"),
    (8,  "src/08_download_france_openmedic.py",   "Download AMELI France Open Medic data"),
    (9,  "src/09_shap_analysis.py",              "SHAP explainability analysis"),
    (10, "src/10_walk_forward_validation.py",     "Walk-forward cross-validation (48 folds)"),
    (11, "src/11_sarima_baseline.py",            "SARIMA baseline model + Ljung-Box test"),
    (12, "src/12_inventory_simulation.py",        "Inventory simulation (s,Q) policy"),
    (13, "src/13_southern_hemisphere_analysis.py","Multi-country Southern Hemisphere validation"),
    (14, "src/14_eda_figures.py",                "EDA visualisations for thesis"),
    (15, "src/15_r06_model.py",                  "R06 model + Diebold-Mariano comparison"),
    (16, "src/16_ensemble_model.py",             "Multi-country SH ensemble (AU+NZ+Chile)"),
    (17, "src/17_switching_ci.py",               "Seasonal switching rule + CI calibration"),
    (18, "src/18_prophet_model.py",              "Prophet baseline + XGBoost vs Prophet DM test"),
    # ── TFG v2 — Nuevas fuentes de datos ─────────────────────────────────────
    (19, "src/19_google_trends.py",              "Google Trends respiratory search signals (ES/EU)"),
    (20, "src/20_weather_signals.py",            "Temperature & humidity signals (Open-Meteo, 7 cities)"),
    (21, "src/05b_integrate_v2.py",              "Integrate external signals → dataset_v2.csv"),
    # ── TFG v2 — Mejoras del modelo ───────────────────────────────────────────
    (22, "src/07b_model_lightgbm.py",            "LightGBM Model A/B/C + DM test vs XGBoost"),
    (23, "src/16b_stacking_ensemble.py",         "Stacking ensemble (Ridge meta-learner, XGB+LGB)"),
    (24, "src/17b_conformal_prediction.py",      "Conformal prediction CI calibration (target 80%)"),
    # ── TFG v3 — Mejoras al modelo XGBoost + nuevos targets ──────────────────
    (26, "src/07c_model_xgboost_v3.py",         "XGBoost v3 enhanced features (cyclical, multi-lag, N02BE cross)"),
    (27, "src/21_n02be_model.py",               "N02BE (paracetamol) demand forecasting — same AU flu framework"),
    # ── Sync final ────────────────────────────────────────────────────────────
    (28, "src/sync_to_notion.py",               "Sync all metrics to Notion (requires NOTION_TOKEN)"),
]

BAR = "=" * 65


def run_step(num, script, description):
    print(f"\n{BAR}")
    print(f"  STEP {num:02d}: {description}")
    print(f"  Script : {script}")
    print(BAR)
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script],
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[FAIL] Step {num} failed after {elapsed:.1f}s — aborting pipeline.")
        sys.exit(1)
    print(f"\n[OK] Step {num} completed in {elapsed:.1f}s")


def main():
    args = sys.argv[1:]
    start_from = 1
    only_step  = None

    i = 0
    while i < len(args):
        if args[i] == "--from" and i + 1 < len(args):
            start_from = int(args[i + 1]); i += 2
        elif args[i] == "--only" and i + 1 < len(args):
            only_step = int(args[i + 1]); i += 2
        else:
            i += 1

    print(BAR)
    print("  PHARMACEUTICAL DEMAND FORECASTING — FULL PIPELINE")
    print(BAR)
    print("  Steps available:")
    for num, script, desc in STEPS:
        print(f"    {num:02d}. {desc}")
    print()

    t_total = time.time()
    for num, script, description in STEPS:
        if only_step is not None and num != only_step:
            continue
        if only_step is None and num < start_from:
            print(f"  [SKIP] Step {num:02d}: {description}")
            continue
        if not os.path.exists(script):
            print(f"  [SKIP] Step {num:02d}: {script} not found")
            continue
        run_step(num, script, description)

    elapsed_total = time.time() - t_total
    print(f"\n{BAR}")
    print(f"  PIPELINE COMPLETE in {elapsed_total:.1f}s")
    print(f"  Outputs in: output/")
    print(f"  Dashboard:  python -m streamlit run dashboard/app.py")
    print(f"  Notion:     python src/sync_to_notion.py  (requiere .env con NOTION_TOKEN)")
    print(BAR)


if __name__ == "__main__":
    main()
