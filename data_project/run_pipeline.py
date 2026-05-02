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

    print(f"\n{BAR}")
    print(f"  PIPELINE COMPLETE in {time.time() - t_total:.1f}s")
    print(f"  Outputs in: output/")
    print(f"  Dashboard:  streamlit run dashboard/app.py")
    print(BAR)


if __name__ == "__main__":
    main()
