"""
Step 4: Clean Australian PBS pharmaceutical data.
Joins PBS Date of Supply CSVs with the item-to-ATC drug map.
Filters for ATC codes R03* (respiratory) and R06* (antihistamines).
Aggregates to monthly prescriptions per ATC group.
Output: processed/australian_pharma_monthly.csv

Run: python src/04_clean_pbs_australia.py
"""

import os
import glob
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(PROC_DIR, exist_ok=True)

PBS_FILES = [
    "pbs_2020_2021.csv",
    "pbs_2021_2022.csv",
    "pbs_2022_2023.csv",
    "pbs_2023_2024.csv",
]


def load_drug_map():
    path = os.path.join(RAW_DIR, "pbs_item_drug_map.csv")
    dm = pd.read_csv(path, encoding="latin-1", usecols=["ITEM_CODE", "DRUG_NAME", "ATC5_Code"])
    dm["ATC5_Code"] = dm["ATC5_Code"].astype(str).str.strip()
    # Keep only R03 and R06 items
    mask = dm["ATC5_Code"].str.startswith("R03") | dm["ATC5_Code"].str.startswith("R06")
    respiratory_items = dm[mask].copy()
    respiratory_items["atc_group"] = respiratory_items["ATC5_Code"].str[:3]  # R03 or R06
    print(f"[OK] Drug map loaded: {len(respiratory_items)} R03/R06 item codes found")
    print(f"     R03 items: {(respiratory_items['atc_group'] == 'R03').sum()}")
    print(f"     R06 items: {(respiratory_items['atc_group'] == 'R06').sum()}")
    return respiratory_items


def load_pbs_files(item_codes):
    dfs = []
    for filename in PBS_FILES:
        path = os.path.join(RAW_DIR, filename)
        if not os.path.exists(path):
            print(f"[SKIP] {filename} not found")
            continue
        print(f"[LOADING] {filename}...")
        df = pd.read_csv(
            path,
            usecols=["MONTH_OF_SUPPLY", "ITEM_CODE", "PRESCRIPTIONS"],
            dtype={"ITEM_CODE": str, "MONTH_OF_SUPPLY": str},
        )
        # Filter to only R03/R06 items immediately to save memory
        df = df[df["ITEM_CODE"].isin(item_codes)]
        dfs.append(df)
        print(f"  > {len(df):,} R03/R06 rows")

    if not dfs:
        raise FileNotFoundError("No PBS files found. Run step 01 first.")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"[OK] Combined: {len(combined):,} rows across all years")
    return combined


def main():
    print("=" * 60)
    print("STEP 4: CLEAN AUSTRALIAN PBS DATA")
    print("=" * 60)

    # 1. Load drug map and get R03/R06 item codes
    drug_map = load_drug_map()
    respiratory_items = set(drug_map["ITEM_CODE"].astype(str))

    # 2. Load and filter PBS prescription data
    pbs = load_pbs_files(respiratory_items)

    # 3. Join with drug map to get ATC group
    pbs = pbs.merge(
        drug_map[["ITEM_CODE", "atc_group", "DRUG_NAME"]],
        on="ITEM_CODE",
        how="left",
    )

    # 4. Parse date (format: YYYYMM)
    pbs["month_date"] = pd.to_datetime(pbs["MONTH_OF_SUPPLY"], format="%Y%m")

    # 5. Aggregate by month and ATC group
    monthly = (
        pbs.groupby(["month_date", "atc_group"])["PRESCRIPTIONS"]
        .sum()
        .reset_index()
    )

    # 6. Pivot to wide format: one column per ATC group
    pivoted = monthly.pivot(index="month_date", columns="atc_group", values="PRESCRIPTIONS")
    pivoted.columns = [f"au_{c}_prescriptions" for c in pivoted.columns]
    pivoted = pivoted.sort_index()

    print(f"\n[INFO] Date range: {pivoted.index.min().date()} to {pivoted.index.max().date()}")
    print(f"[INFO] Months: {len(pivoted)}")
    print(f"[INFO] Columns: {list(pivoted.columns)}")

    # 7. Summary statistics
    print("\n--- MONTHLY PRESCRIPTIONS SUMMARY (Australia) ---")
    print(pivoted.describe().round(0))

    # 8. Quick seasonal check
    pivoted["month"] = pivoted.index.month
    if "au_R03_prescriptions" in pivoted.columns:
        peak = pivoted.groupby("month")["au_R03_prescriptions"].mean().idxmax()
        print(f"\n[VALIDATION] R03 peak prescription month in Australia: {peak}")
        print(f"             (Expected: June-August = months 6-8 -> winter/flu season)")
    pivoted.drop(columns=["month"], inplace=True)

    # 9. Save
    out_path = os.path.join(PROC_DIR, "australian_pharma_monthly.csv")
    pivoted.reset_index().to_csv(out_path, index=False)
    print(f"\n[OK] Saved {out_path} ({len(pivoted)} rows)")
    print("[DONE] Australian PBS processing complete.")


if __name__ == "__main__":
    main()
