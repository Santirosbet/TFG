"""
Step 5: Integrate all datasets into a single analysis-ready table.
Merges: European sales (weekly) + Australian pharma (monthly→weekly) + FluNet (weekly).
Applies the REAL 26-week lag from Australian data.
Output: processed/integrated_dataset.csv

Run: python src/05_integrate_datasets.py
"""

import os
import pandas as pd
import numpy as np

PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

LEAD_WEEKS = 26


def load_european_sales():
    path = os.path.join(PROC_DIR, "european_pharma_weekly.csv")
    df = pd.read_csv(path, parse_dates=["week_date"], index_col="week_date")
    print(f"[OK] European sales: {len(df)} weeks ({df.index.min().date()} to {df.index.max().date()})")
    return df


def load_flunet(hemisphere):
    path = os.path.join(PROC_DIR, f"flunet_{hemisphere}.csv")
    if not os.path.exists(path):
        print(f"[SKIP] FluNet {hemisphere} not found at {path}")
        return None
    df = pd.read_csv(path, parse_dates=["iso_date"])
    df = df.set_index("iso_date").sort_index()
    print(f"[OK] FluNet {hemisphere}: {len(df)} weeks")
    return df


def load_australian_pharma():
    path = os.path.join(PROC_DIR, "australian_pharma_monthly.csv")
    if not os.path.exists(path):
        print(f"[SKIP] Australian pharma data not found at {path}")
        return None
    df = pd.read_csv(path, parse_dates=["month_date"])
    df = df.set_index("month_date").sort_index()
    print(f"[OK] Australian pharma: {len(df)} months")
    return df


def resample_monthly_to_weekly(monthly_df):
    """Upsample monthly data to weekly using forward-fill."""
    weekly = monthly_df.resample("W-SUN").ffill()
    return weekly


def apply_real_lag(au_series, weeks=LEAD_WEEKS):
    """
    Apply the REAL hemispheric lag.
    Australian data from week T is used to predict European data at week T+26.
    This means we shift the Australian series FORWARD by 26 weeks.
    """
    return au_series.shift(weeks)


def main():
    print("=" * 60)
    print("STEP 5: INTEGRATE ALL DATASETS")
    print("=" * 60)

    eu_sales = load_european_sales()

    # Start with the European base
    integrated = eu_sales.copy()

    # --- Add FluNet epidemiological data ---
    flu_au = load_flunet("australia")
    flu_eu = load_flunet("europe")

    if flu_au is not None and "INF_ALL" in flu_au.columns:
        flu_au_weekly = flu_au[["INF_ALL"]].rename(columns={"INF_ALL": "flu_au_positives"})
        flu_au_weekly = flu_au_weekly.resample("W-SUN").sum()
        # Apply the real 26-week lag: Australian flu data LEADS European demand
        flu_au_weekly["flu_au_lagged"] = apply_real_lag(flu_au_weekly["flu_au_positives"], LEAD_WEEKS)
        integrated = integrated.join(flu_au_weekly, how="left")
        print(f"[OK] Merged Australian flu data (with {LEAD_WEEKS}-week lag)")

    if flu_eu is not None and "INF_ALL" in flu_eu.columns:
        flu_eu_weekly = flu_eu[["INF_ALL"]].rename(columns={"INF_ALL": "flu_eu_positives"})
        flu_eu_weekly = flu_eu_weekly.resample("W-SUN").sum()
        integrated = integrated.join(flu_eu_weekly, how="left")
        print("[OK] Merged European flu data")

    # --- Add Australian pharmaceutical demand (if available) ---
    au_pharma = load_australian_pharma()
    if au_pharma is not None:
        au_weekly = resample_monthly_to_weekly(au_pharma)
        # Apply the real lag
        for col in au_weekly.columns:
            au_weekly[f"{col}_lagged"] = apply_real_lag(au_weekly[col], LEAD_WEEKS)
        # Drop the unlagged columns, keep only lagged
        lagged_cols = [c for c in au_weekly.columns if c.endswith("_lagged")]

        # Check for date overlap before merging
        overlap = integrated.index.intersection(au_weekly.index)
        if len(overlap) == 0:
            print(f"[WARNING] PBS data ({au_weekly.index.min().date()} to {au_weekly.index.max().date()})")
            print(f"          does not overlap with European data ({integrated.index.min().date()} to {integrated.index.max().date()})")
            print(f"          Skipping PBS merge for the model. PBS data saved separately for validation.")
            au_weekly[lagged_cols].to_csv(
                os.path.join(PROC_DIR, "australian_pharma_weekly_lagged.csv")
            )
        else:
            integrated = integrated.join(au_weekly[lagged_cols], how="left")
            print(f"[OK] Merged Australian pharma data (with {LEAD_WEEKS}-week lag)")

    # --- Feature engineering: autoregressive lag features ---
    integrated["R03_lag1"] = integrated["R03"].shift(1)
    integrated["R03_lag4_avg"] = integrated["R03"].shift(1).rolling(window=4).mean()
    integrated["R06_lag1"] = integrated["R06"].shift(1)
    print("[OK] Created autoregressive lag features (lag1, lag4_avg)")

    # --- Clean: drop rows with NaN from lag operations ---
    before = len(integrated)
    integrated = integrated.dropna()
    after = len(integrated)
    print(f"[INFO] Dropped {before - after} rows with NaN values (from lag operations)")

    # --- Save ---
    out_path = os.path.join(PROC_DIR, "integrated_dataset.csv")
    integrated.to_csv(out_path)
    print(f"\n[OK] Saved {out_path}")
    print(f"     Shape: {integrated.shape}")
    print(f"     Date range: {integrated.index.min().date()} to {integrated.index.max().date()}")
    print(f"     Columns: {list(integrated.columns)}")

    # --- Summary statistics ---
    print("\n--- INTEGRATED DATASET SUMMARY ---")
    print(integrated.describe().round(2))

    print("\n[DONE] Integration complete.")


if __name__ == "__main__":
    main()
