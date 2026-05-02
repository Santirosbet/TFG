"""
Step 6: Validate the lead-lag hypothesis using REAL epidemiological data.
Computes cross-correlation between Australian and European flu seasons.
This replaces the synthetic proxy validation with real evidence.
Output: output/lead_lag_validation.png, console report

Run: python src/06_validate_lead_lag.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def cross_correlation(series_a, series_b, max_lag=52):
    """
    Compute cross-correlation between two series at different lags.
    Returns dict of {lag_weeks: correlation_coefficient}.
    """
    results = {}
    a = series_a.values
    b = series_b.values

    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            corr = np.corrcoef(a[lag:], b[:-lag])[0, 1]
        elif lag < 0:
            corr = np.corrcoef(a[:lag], b[-lag:])[0, 1]
        else:
            corr = np.corrcoef(a, b)[0, 1]

        if not np.isnan(corr):
            results[lag] = corr

    return results


def main():
    print("=" * 60)
    print("STEP 6: VALIDATE LEAD-LAG HYPOTHESIS (REAL DATA)")
    print("=" * 60)

    # Load processed FluNet data
    au_path = os.path.join(PROC_DIR, "flunet_australia.csv")
    eu_path = os.path.join(PROC_DIR, "flunet_europe.csv")

    if not os.path.exists(au_path) or not os.path.exists(eu_path):
        print("[ERROR] FluNet processed data not found. Run steps 01 and 02 first.")
        return

    au = pd.read_csv(au_path, parse_dates=["iso_date"]).set_index("iso_date").sort_index()
    eu = pd.read_csv(eu_path, parse_dates=["iso_date"]).set_index("iso_date").sort_index()

    if "INF_ALL" not in au.columns or "INF_ALL" not in eu.columns:
        print("[ERROR] INF_ALL column not found in FluNet data.")
        print(f"  Australia columns: {list(au.columns)}")
        print(f"  Europe columns: {list(eu.columns)}")
        return

    # Align the two series to the same date range
    common_start = max(au.index.min(), eu.index.min())
    common_end = min(au.index.max(), eu.index.max())
    au_aligned = au.loc[common_start:common_end, "INF_ALL"].resample("W-SUN").sum()
    eu_aligned = eu.loc[common_start:common_end, "INF_ALL"].resample("W-SUN").sum()

    # Align indices
    common_idx = au_aligned.index.intersection(eu_aligned.index)
    au_aligned = au_aligned.loc[common_idx]
    eu_aligned = eu_aligned.loc[common_idx]

    print(f"[INFO] Aligned series: {len(common_idx)} weeks")
    print(f"       Range: {common_idx.min().date()} to {common_idx.max().date()}")

    # --- 1. Pearson correlation (no lag) ---
    pearson_r = np.corrcoef(au_aligned.values, eu_aligned.values)[0, 1]
    print(f"\n[RESULT] Pearson correlation (lag=0): {pearson_r:.4f}")
    print("         (This should be NEGATIVE or near-zero, confirming opposite seasons)")

    # --- 2. Cross-correlation at multiple lags ---
    print("\n[COMPUTING] Cross-correlation at lags 0-52 weeks...")
    cc = cross_correlation(au_aligned, eu_aligned, max_lag=52)

    best_lag = max(cc, key=cc.get)
    best_corr = cc[best_lag]
    print(f"[RESULT] Maximum correlation: {best_corr:.4f} at lag = {best_lag} weeks")
    print(f"         This means Australian flu LEADS European flu by ~{abs(best_lag)} weeks")

    # Show correlation at key lags
    print("\n--- Cross-correlation at key lags ---")
    for lag in [0, 13, 20, 22, 24, 26, 28, 30, 39, 52]:
        if lag in cc:
            print(f"  Lag {lag:3d} weeks: r = {cc[lag]:.4f}")

    # --- 3. Visualization ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Overlay of flu seasons
    ax1 = axes[0]
    ax1.plot(au_aligned.index, au_aligned.values, label="Australia (influenza positives)", color="blue", alpha=0.7)
    ax1.plot(eu_aligned.index, eu_aligned.values, label="Europe (influenza positives)", color="red", alpha=0.7)
    ax1.set_title("Weekly Influenza Positives: Australia vs Europe (WHO FluNet)", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Influenza Positives")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Cross-correlation function
    ax2 = axes[1]
    lags = sorted(cc.keys())
    corrs = [cc[l] for l in lags]
    ax2.bar(lags, corrs, color=["green" if l == best_lag else "steelblue" for l in lags], alpha=0.7)
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax2.axvline(x=26, color="red", linestyle="--", alpha=0.5, label="26-week hypothesis")
    ax2.axvline(x=best_lag, color="green", linestyle="--", alpha=0.5, label=f"Best lag: {best_lag} weeks")
    ax2.set_title("Cross-Correlation: Australian Flu → European Flu", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Lag (weeks, positive = Australia leads)")
    ax2.set_ylabel("Pearson r")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "lead_lag_validation.png")
    plt.savefig(fig_path, dpi=150)
    print(f"\n[OK] Saved visualization: {fig_path}")
    plt.close()

    # --- 4. Monthly peak comparison ---
    print("\n--- MONTHLY PEAK ANALYSIS ---")
    au_monthly = au_aligned.copy()
    au_monthly.index = au_aligned.index
    au_by_month = au_aligned.groupby(au_aligned.index.month).mean()
    eu_by_month = eu_aligned.groupby(eu_aligned.index.month).mean()

    print(f"  Australia peak month: {au_by_month.idxmax()} (avg: {au_by_month.max():.0f})")
    print(f"  Europe peak month:    {eu_by_month.idxmax()} (avg: {eu_by_month.max():.0f})")
    peak_diff = (eu_by_month.idxmax() - au_by_month.idxmax()) % 12
    print(f"  Peak difference:      {peak_diff} months (~{peak_diff * 4} weeks)")

    print("\n[DONE] Lead-lag validation complete.")
    print("\nCONCLUSION FOR THESIS:")
    if abs(best_lag - 26) <= 6:
        print(f"  The cross-correlation analysis CONFIRMS the ~26-week lead-lag hypothesis.")
        print(f"  Optimal lag found: {best_lag} weeks (r={best_corr:.4f}).")
        print(f"  This provides empirical evidence from real WHO data, not synthetic proxies.")
    else:
        print(f"  The optimal lag ({best_lag} weeks) differs from the 26-week hypothesis.")
        print(f"  Consider adjusting the proxy lag or discussing this finding in the thesis.")


if __name__ == "__main__":
    main()
