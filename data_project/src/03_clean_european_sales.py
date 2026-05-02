"""
Step 3: Clean the Kaggle European pharmacy sales dataset.
Extracts R03 and R06 weekly series. Validates temporal continuity.
Output: processed/european_pharma_weekly.csv

Run: python src/03_clean_european_sales.py
"""

import os
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(PROC_DIR, exist_ok=True)


def main():
    print("=" * 60)
    print("STEP 3: CLEAN EUROPEAN PHARMACY SALES (KAGGLE)")
    print("=" * 60)

    path = os.path.join(RAW_DIR, "salesweekly.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"salesweekly.csv not found at {path}. Run step 01 first.")

    df = pd.read_csv(path)
    print(f"[OK] Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"[INFO] Columns: {list(df.columns)}")

    df["datum"] = pd.to_datetime(df["datum"])
    df.set_index("datum", inplace=True)
    df = df.sort_index()

    # Validate temporal continuity
    expected_weeks = pd.date_range(start=df.index.min(), end=df.index.max(), freq="W-SUN")
    missing_weeks = len(expected_weeks) - len(df)
    print(f"[INFO] Date range: {df.index.min().date()} to {df.index.max().date()}")
    print(f"[INFO] Total weeks: {len(df)}")
    print(f"[INFO] Missing weeks: {missing_weeks}")

    # Validate target columns
    for col in ["R03", "R06"]:
        nulls = df[col].isnull().sum()
        print(f"[INFO] {col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}, "
              f"min={df[col].min():.0f}, max={df[col].max():.0f}, nulls={nulls}")

    # Extract only the columns we need
    output = df[["R03", "R06"]].copy()
    output.index.name = "week_date"

    out_path = os.path.join(PROC_DIR, "european_pharma_weekly.csv")
    output.to_csv(out_path)
    print(f"\n[OK] Saved {out_path} ({len(output)} rows)")
    print("[DONE] European sales processing complete.")


if __name__ == "__main__":
    main()
