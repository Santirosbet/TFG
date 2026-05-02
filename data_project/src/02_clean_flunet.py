"""
Step 2: Clean and process WHO FluNet data.
Extracts weekly influenza positives for Australia and European countries.
Outputs: processed/flunet_australia.csv, processed/flunet_europe.csv

Run: python src/02_clean_flunet.py
"""

import os
import pandas as pd
import numpy as np

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(PROC_DIR, exist_ok=True)

EUROPEAN_COUNTRIES = [
    "Spain", "France", "Germany", "Italy", "United Kingdom of Great Britain and Northern Ireland",
    "Netherlands", "Belgium", "Portugal", "Austria", "Sweden",
    "Denmark", "Finland", "Norway", "Ireland", "Switzerland",
    "Poland", "Czechia", "Greece", "Hungary", "Romania",
]

AUSTRALIA = "Australia"


def load_flunet():
    path = os.path.join(RAW_DIR, "who_flunet_global.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"FluNet data not found at {path}. Run 01_download_data.py first."
        )
    print(f"[LOADING] {path}...")
    df = pd.read_csv(path, low_memory=False)
    print(f"[OK] Loaded {len(df):,} rows, {len(df.columns)} columns")
    print(f"[INFO] Columns: {list(df.columns[:15])}...")
    return df


def process_country_group(df, countries, group_name):
    """Filter FluNet for a list of countries, aggregate by ISO week."""
    mask = df["COUNTRY_AREA_TERRITORY"].isin(countries) if isinstance(countries, list) else df["COUNTRY_AREA_TERRITORY"] == countries
    subset = df[mask].copy()

    if subset.empty:
        print(f"[WARNING] No data found for {group_name}. Check country names.")
        print(f"  Available countries sample: {df['COUNTRY_AREA_TERRITORY'].unique()[:20]}")
        return pd.DataFrame()

    print(f"[INFO] {group_name}: {len(subset):,} rows, countries: {subset['COUNTRY_AREA_TERRITORY'].nunique()}")

    numeric_cols = ["INF_ALL", "INF_A", "INF_B", "SPEC_PROCESSED_NB", "SPEC_RECEIVED_NB"]
    available_cols = [c for c in numeric_cols if c in subset.columns]

    for col in available_cols:
        subset[col] = pd.to_numeric(subset[col], errors="coerce")

    if "ISO_YEAR" in subset.columns and "ISO_WEEK" in subset.columns:
        year_col, week_col = "ISO_YEAR", "ISO_WEEK"
    elif "MMWR_YEAR" in subset.columns:
        year_col, week_col = "MMWR_YEAR", "MMWR_WEEKSTARTDATE"
    else:
        year_col = [c for c in subset.columns if "YEAR" in c.upper()][0]
        week_col = [c for c in subset.columns if "WEEK" in c.upper()][0]

    grouped = (
        subset.groupby([year_col, week_col])[available_cols]
        .sum()
        .reset_index()
    )

    grouped.rename(columns={year_col: "year", week_col: "week"}, inplace=True)
    grouped["iso_date"] = pd.to_datetime(
        grouped["year"].astype(str) + "-W" + grouped["week"].astype(str).str.zfill(2) + "-1",
        format="%G-W%V-%u",
        errors="coerce",
    )
    grouped = grouped.dropna(subset=["iso_date"]).sort_values("iso_date")
    grouped["group"] = group_name

    return grouped


def main():
    print("=" * 60)
    print("STEP 2: CLEAN WHO FluNet DATA")
    print("=" * 60)

    df = load_flunet()

    # Process Australia
    au = process_country_group(df, AUSTRALIA, "Australia")
    if not au.empty:
        au_path = os.path.join(PROC_DIR, "flunet_australia.csv")
        au.to_csv(au_path, index=False)
        print(f"[OK] Saved {au_path} ({len(au)} rows)")
        print(f"     Date range: {au['iso_date'].min()} to {au['iso_date'].max()}")

    # Process Europe (aggregated)
    eu = process_country_group(df, EUROPEAN_COUNTRIES, "Europe")
    if not eu.empty:
        eu_path = os.path.join(PROC_DIR, "flunet_europe.csv")
        eu.to_csv(eu_path, index=False)
        print(f"[OK] Saved {eu_path} ({len(eu)} rows)")
        print(f"     Date range: {eu['iso_date'].min()} to {eu['iso_date'].max()}")

    # Quick validation: show peak months for each hemisphere
    print()
    print("--- SEASONAL VALIDATION ---")
    for name, data in [("Australia", au), ("Europe", eu)]:
        if data.empty:
            continue
        data["month"] = pd.to_datetime(data["iso_date"]).dt.month
        if "INF_ALL" in data.columns:
            monthly = data.groupby("month")["INF_ALL"].mean()
            peak_month = monthly.idxmax()
            print(f"  {name} peak influenza month: {peak_month} (avg positives: {monthly.max():.0f})")

    print("\n[DONE] FluNet processing complete.")


if __name__ == "__main__":
    main()
