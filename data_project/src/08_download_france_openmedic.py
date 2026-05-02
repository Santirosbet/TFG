"""
Step 8: Download French Open Medic data from AMELI (Assurance Maladie).
Source: https://data.gouv.fr - Open Medic ATC3 annual data
Filters for R03 (respiratory) and R06 (antihistamines) - exact same ATC classification as thesis.
Output: data/raw/france_openmedic_atc3.csv
         data/processed/france_r03_r06_annual.csv

Run: python src/08_download_france_openmedic.py
"""

import os
import re
import gzip
import io
import requests
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)

BASE_URL = "https://open-data-assurance-maladie.ameli.fr/medicaments/"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; academic-research/1.0)"}
YEARS = list(range(2014, 2025))  # 2014-2024


def get_atc3_download_url(year: int) -> str | None:
    """Get the download URL for the NB_YYYY_atc3.CSV.gz file."""
    page_url = f"{BASE_URL}download2.php?Dir_Rep={year}_ATC3"
    try:
        r = requests.get(page_url, headers=HEADERS, timeout=20)
        if r.status_code != 200:
            return None
        links = re.findall(r'href="(./download_file\.php[^"]+)"', r.text)
        # First link is always the main NB_YYYY_atc3.CSV.gz
        for link in links:
            if f"NB_{year}_atc3.CSV.gz" in link or f"NB_{year}_atc3.CSV" in link:
                return BASE_URL + link.replace("./", "")
        if links:
            return BASE_URL + links[0].replace("./", "")
        return None
    except Exception as e:
        print(f"[ERROR] Could not get URL for {year}: {e}")
        return None


def download_atc3_year(year: int) -> pd.DataFrame | None:
    """Download and parse ATC3 data for a given year."""
    url = get_atc3_download_url(year)
    if not url:
        print(f"[SKIP] {year} - no URL found")
        return None
    try:
        r = requests.get(url, headers=HEADERS, timeout=40)
        if r.status_code != 200:
            print(f"[SKIP] {year} - HTTP {r.status_code}")
            return None
        raw = gzip.decompress(r.content)
        text = raw.decode("latin-1")
        df = pd.read_csv(io.StringIO(text), sep=";")
        df["year"] = year
        print(f"[OK] {year}: {len(df)} ATC3 rows")
        return df
    except Exception as e:
        print(f"[ERROR] {year}: {e}")
        return None


def clean_french_number(s) -> float:
    """Convert French number format (1.234.567,89) to float."""
    if pd.isna(s):
        return 0.0
    s = str(s).replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return 0.0


def main():
    print("=" * 60)
    print("STEP 8: DOWNLOAD FRENCH OPEN MEDIC (AMELI) DATA")
    print("Source: Assurance Maladie - data.gouv.fr")
    print("=" * 60)

    all_dfs = []
    for year in YEARS:
        df = download_atc3_year(year)
        if df is not None:
            all_dfs.append(df)

    if not all_dfs:
        print("[ERROR] No data downloaded. Check network connection.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)

    # Save raw combined
    raw_path = os.path.join(RAW_DIR, "france_openmedic_atc3.csv")
    combined.to_csv(raw_path, index=False)
    print(f"\n[OK] Raw data saved: {raw_path} ({len(combined)} rows)")

    # Filter for R03 and R06
    atc_col = combined.columns[0]  # ATC3 column name varies slightly
    r_mask = combined[atc_col].astype(str).str.startswith("R03") | \
             combined[atc_col].astype(str).str.startswith("R06")
    respiratory = combined[r_mask].copy()
    respiratory["atc_group"] = respiratory[atc_col].astype(str).str[:3]

    # Clean BOITES (boxes dispensed) column
    if "BOITES" in respiratory.columns:
        respiratory["boites_clean"] = respiratory["BOITES"].apply(clean_french_number)
    elif "BOI" in " ".join(respiratory.columns):
        boites_col = [c for c in respiratory.columns if "BOI" in c.upper()][0]
        respiratory["boites_clean"] = respiratory[boites_col].apply(clean_french_number)
    else:
        # Use last numeric column
        respiratory["boites_clean"] = respiratory.iloc[:, -2].apply(clean_french_number)

    # Aggregate per year and ATC group
    agg = (
        respiratory.groupby(["year", "atc_group"])["boites_clean"]
        .sum()
        .reset_index()
        .rename(columns={"boites_clean": "france_boxes_dispensed"})
    )

    # Pivot to wide format
    pivot = agg.pivot(index="year", columns="atc_group", values="france_boxes_dispensed")
    pivot.columns = [f"france_{c}_boxes" for c in pivot.columns]
    pivot = pivot.reset_index()

    print("\n--- FRANCE R03/R06 ANNUAL DATA ---")
    print(pivot.to_string(index=False))

    # Save processed
    proc_path = os.path.join(PROC_DIR, "france_r03_r06_annual.csv")
    pivot.to_csv(proc_path, index=False)
    print(f"\n[OK] Processed data saved: {proc_path}")
    print(f"     Years: {pivot['year'].min()} - {pivot['year'].max()}")
    print(f"     Columns: {list(pivot.columns)}")

    # Validation: confirm seasonal pattern (R03 should peak in winter = implies higher annual totals in flu years)
    if "france_R03_boxes" in pivot.columns:
        avg = pivot["france_R03_boxes"].mean() / 1e6
        print(f"\n[VALIDATION] France R03 avg dispensed: {avg:.1f}M boxes/year")
        print(f"             (French population: 68M -> ~{avg/68:.1f}M boxes per million inhabitants)")

    print("\n[DONE] French Open Medic download complete.")


if __name__ == "__main__":
    main()
