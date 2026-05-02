"""
Step 1: Download all real datasets for the TFG.
Sources: WHO FluNet, Kaggle (already local), PBS/AIHW (manual download instructions).

Run: python src/01_download_data.py
"""

import os
import requests
import shutil

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)


def download_file(url, filename, description):
    filepath = os.path.join(RAW_DIR, filename)
    if os.path.exists(filepath):
        print(f"[SKIP] {filename} already exists.")
        return filepath

    print(f"[DOWNLOADING] {description}...")
    print(f"  URL: {url}")
    try:
        resp = requests.get(url, timeout=120, stream=True)
        resp.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"[OK] Saved {filename} ({size_mb:.1f} MB)")
        return filepath
    except Exception as e:
        print(f"[ERROR] Failed to download {filename}: {e}")
        return None


def copy_kaggle_dataset():
    """Copy the local Kaggle CSV (European pharmacy sales) into the raw data folder."""
    src = os.path.join(os.path.dirname(__file__), "..", "..", "salesweekly.csv")
    dst = os.path.join(RAW_DIR, "salesweekly.csv")
    if os.path.exists(dst):
        print("[SKIP] salesweekly.csv already in raw folder.")
        return
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print("[OK] Copied salesweekly.csv to data/raw/")
    else:
        print("[WARNING] salesweekly.csv not found at expected path. Copy it manually.")


def main():
    print("=" * 60)
    print("STEP 1: DATA ACQUISITION")
    print("=" * 60)

    # --- 1. WHO FluNet (auto-download, CSV, no login required) ---
    download_file(
        url="https://xmart-api-public.who.int/FLUMART/VIW_FNT?$format=csv",
        filename="who_flunet_global.csv",
        description="WHO FluNet — Global influenza virological data (1997-present)",
    )

    download_file(
        url="https://xmart-api-public.who.int/FLUMART/VIW_FLU_METADATA?$format=csv",
        filename="who_flunet_metadata.csv",
        description="WHO FluNet — Data dictionary / metadata",
    )

    # --- 2. Kaggle European pharmacy data (local copy) ---
    copy_kaggle_dataset()

    # --- 3. Manual download instructions for PBS/AIHW ---
    print()
    print("=" * 60)
    print("MANUAL DOWNLOADS REQUIRED (government portals):")
    print("=" * 60)
    print()
    print("3a. PBS Date of Supply Data (Australian pharmaceutical prescriptions)")
    print("    URL: https://www.pbs.gov.au/info/statistics/dos-and-dop/dos-and-dop")
    print("    Download: 'Section 85 Date of Supply' Excel file")
    print("    Save as: data/raw/pbs_date_of_supply.xlsx")
    print()
    print("3b. AIHW PBS Monthly Prescriptions")
    print("    URL: https://www.aihw.gov.au/reports/medicines/pbs-monthly-data/data")
    print("    Download: Data tables (Excel)")
    print("    Save as: data/raw/aihw_pbs_monthly.xlsx")
    print()
    print("3c. PBS ATC Historical Report (1992-2016)")
    print("    URL: https://data.gov.au/data/dataset/pharmaceutical-benefits-scheme-pbs-anatomical-therapeutic-chemical-report")
    print("    Download: CSV (zipped)")
    print("    Save as: data/raw/pbs_atc_historical.csv")
    print()
    print("3d. NNDSS Influenza (Australian lab-confirmed cases)")
    print("    URL: https://www.cdc.gov.au/resources/publications/nndss-dataset-influenza")
    print("    Download: Excel file")
    print("    Save as: data/raw/nndss_influenza_australia.xlsx")
    print()
    print("3e. ECDC Respiratory Viruses (European flu data)")
    print("    URL: https://github.com/EU-ECDC/Respiratory_viruses_weekly_data")
    print("    Download: ILIARIRates.csv from the repository")
    print("    Save as: data/raw/ecdc_ili_rates.csv")
    print()
    print("[DONE] Automated downloads complete. Please complete manual downloads above.")


if __name__ == "__main__":
    main()
