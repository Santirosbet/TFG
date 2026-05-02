"""
Step 17: Seasonal Switching Rule + CI Calibration.

Part A — Seasonal Switching Rule:
  Implements a hybrid forecast policy that uses XGBoost predictions during
  the active flu season (weeks 1-21 and 40-52) and falls back to the
  historical seasonal mean during the off-season (weeks 22-39), where
  near-zero demand makes percentage errors uninformative and high.
  Evaluated on the 192 WFV predictions.

Part B — CI Calibration Check:
  Verifies that the empirical confidence intervals used in the Calculadora
  Predictiva actually achieve the nominal coverage probability.
  A well-calibrated 80% CI should contain ~80% of actual observations.

Outputs:
  output/switching_rule_results.csv  — week-by-week predictions + rule applied
  output/switching_meta.json         — MAPE improvement summary
  output/switching_plot.png          — switching rule visualisation
  output/ci_calibration.json         — coverage probabilities at 50% and 80%
  output/ci_calibration_plot.png     — reliability diagram

Run: python src/17_switching_ci.py
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "output")

BLUE   = "#1f4e79"
ORANGE = "#ff6b35"
GREEN  = "#70ad47"
RED    = "#c00000"
GREY   = "#aaaaaa"
PURPLE = "#7030a0"

OFF_SEASON_START = 22
OFF_SEASON_END   = 39


def compute_mape(y_true, y_pred, mask=None):
    if mask is not None:
        y_true, y_pred = y_true[mask], y_pred[mask]
    m = y_true != 0
    return float(np.mean(np.abs((y_true[m] - y_pred[m]) / y_true[m])) * 100)


# ══════════════════════════════════════════════════════════════════════════════
# PART A — SEASONAL SWITCHING RULE
# ══════════════════════════════════════════════════════════════════════════════

def switching_rule(wfv_df, df_full):
    wfv = wfv_df.copy()
    wfv["week_date"] = pd.to_datetime(wfv["week_date"])
    wfv["iso_week"]  = wfv["week_date"].dt.isocalendar().week.astype(int)

    # Seasonal mean per ISO week from training data (up to split)
    with open(os.path.join(OUT_DIR, "model_meta.json")) as f:
        meta = json.load(f)
    split_idx = meta["split_idx"]
    train_df = df_full.iloc[:split_idx].copy()
    train_df["iso_week"] = train_df.index.isocalendar().week.astype(int)
    seasonal_mean = train_df.groupby("iso_week")["R03"].mean().to_dict()

    # Apply switching rule
    wfv["seasonal_mean"] = wfv["iso_week"].map(seasonal_mean)
    wfv["in_off_season"] = wfv["iso_week"].between(OFF_SEASON_START, OFF_SEASON_END)
    wfv["pred_switched"] = np.where(
        wfv["in_off_season"],
        wfv["seasonal_mean"],
        wfv["pred_B"]
    )

    # Metrics
    actual = wfv["actual_R03"].values
    pred_b = wfv["pred_B"].values
    pred_sw = wfv["pred_switched"].values

    season_mask     = ~wfv["in_off_season"].values
    offseason_mask  = wfv["in_off_season"].values

    metrics = {
        "model_b_full_mape":      compute_mape(actual, pred_b),
        "switched_full_mape":     compute_mape(actual, pred_sw),
        "model_b_season_mape":    compute_mape(actual, pred_b,  season_mask),
        "switched_season_mape":   compute_mape(actual, pred_sw, season_mask),
        "model_b_offseason_mape": compute_mape(actual, pred_b,  offseason_mask),
        "switched_offseason_mape":compute_mape(actual, pred_sw, offseason_mask),
        "off_season_weeks":       int(offseason_mask.sum()),
        "peak_season_weeks":      int(season_mask.sum()),
        "improvement_pp":         round(compute_mape(actual, pred_b) - compute_mape(actual, pred_sw), 3),
    }

    print("\n  Seasonal switching rule results:")
    print(f"    Full period  : Model B={metrics['model_b_full_mape']:.1f}%  "
          f"Switched={metrics['switched_full_mape']:.1f}%  "
          f"Improvement={metrics['improvement_pp']:+.1f}pp")
    print(f"    Peak season  : Model B={metrics['model_b_season_mape']:.1f}%  "
          f"Switched={metrics['switched_season_mape']:.1f}%")
    print(f"    Off-season   : Model B={metrics['model_b_offseason_mape']:.1f}%  "
          f"Switched={metrics['switched_offseason_mape']:.1f}%")

    wfv.to_csv(os.path.join(OUT_DIR, "switching_rule_results.csv"), index=False)
    print(f"[OK] Saved: output/switching_rule_results.csv")
    return wfv, metrics


def plot_switching(wfv, metrics):
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle("Seasonal Switching Rule — XGBoost + Seasonal Mean Hybrid\n"
                 f"Off-season weeks {OFF_SEASON_START}-{OFF_SEASON_END}: "
                 "replace XGBoost with historical seasonal mean",
                 fontsize=12, fontweight="bold", color=BLUE)

    dates  = pd.to_datetime(wfv["week_date"])
    actual = wfv["actual_R03"].values
    pred_b = wfv["pred_B"].values
    pred_sw= wfv["pred_switched"].values
    off    = wfv["in_off_season"].values

    ax = axes[0]
    ax.plot(dates, actual, color="black", lw=2, label="Actual", zorder=5)
    ax.plot(dates, pred_b,  color=GREY,   lw=1.5, linestyle="--",
            label=f"Model B (MAPE={metrics['model_b_full_mape']:.1f}%)", alpha=0.7)
    ax.plot(dates, pred_sw, color=BLUE,   lw=2,
            label=f"Switched (MAPE={metrics['switched_full_mape']:.1f}%)")

    # shade off-season
    in_off = False
    start  = None
    for i, (d, o) in enumerate(zip(dates, off)):
        if o and not in_off:
            start = d; in_off = True
        elif not o and in_off:
            ax.axvspan(start, d, alpha=0.08, color=ORANGE)
            in_off = False
    if in_off:
        ax.axvspan(start, dates.iloc[-1], alpha=0.08, color=ORANGE)

    ax.set_ylabel("R03 units / week", fontsize=9)
    ax.set_title("Predictions: XGBoost vs Hybrid Switching Rule", fontsize=10, color=BLUE)
    ax.legend(fontsize=8); ax.grid(alpha=0.2, linestyle="--")
    ax.spines[["top","right"]].set_visible(False)

    from matplotlib.patches import Patch
    ax.legend(handles=list(ax.get_lines()) + [Patch(color=ORANGE, alpha=0.15,
              label=f"Off-season (wks {OFF_SEASON_START}-{OFF_SEASON_END})")],
              fontsize=8)

    ax2 = axes[1]
    cats   = ["Full period\n(192 weeks)", f"Peak season\n(wks 1-21, 40-52)",
              f"Off-season\n(wks {OFF_SEASON_START}-{OFF_SEASON_END})"]
    mb_vals = [metrics["model_b_full_mape"], metrics["model_b_season_mape"],
               metrics["model_b_offseason_mape"]]
    sw_vals = [metrics["switched_full_mape"], metrics["switched_season_mape"],
               metrics["switched_offseason_mape"]]
    x = np.arange(len(cats)); w = 0.35
    b1 = ax2.bar(x - w/2, mb_vals, w, color=GREY,  alpha=0.85, label="Model B only",  edgecolor="white")
    b2 = ax2.bar(x + w/2, sw_vals, w, color=BLUE,  alpha=0.85, label="Hybrid switching", edgecolor="white")
    ax2.bar_label(b1, fmt="%.1f%%", padding=3, fontsize=9)
    ax2.bar_label(b2, fmt="%.1f%%", padding=3, fontsize=9)
    ax2.set_xticks(x); ax2.set_xticklabels(cats, fontsize=10)
    ax2.set_ylabel("MAPE (%)", fontsize=9)
    ax2.set_title("MAPE by season: Model B vs Hybrid", fontsize=10, color=BLUE)
    ax2.legend(fontsize=9); ax2.grid(axis="y", alpha=0.2, linestyle="--")
    ax2.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "switching_plot.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# PART B — CI CALIBRATION CHECK
# ══════════════════════════════════════════════════════════════════════════════

def ci_calibration(wfv_df):
    wfv = wfv_df.copy()
    wfv["week_date"] = pd.to_datetime(wfv["week_date"])
    wfv["iso_week"]  = wfv["week_date"].dt.isocalendar().week.astype(int)

    # Compute the same error profile used in the dashboard (p10/p50/p90 per week)
    wfv["abs_rel_err"] = np.abs(
        (wfv["actual_R03"] - wfv["pred_B"]) / wfv["actual_R03"].replace(0, np.nan)
    )
    err_profile = wfv.groupby("iso_week")["abs_rel_err"].quantile([0.10, 0.50, 0.90]).unstack()
    err_profile.columns = ["p10", "p50", "p90"]

    # For each prediction, compute CI bounds and check coverage
    wfv = wfv.merge(err_profile.reset_index(), on="iso_week", how="left")
    wfv["ci_lo_80"] = wfv["pred_B"] * (1 - wfv["p90"])
    wfv["ci_hi_80"] = wfv["pred_B"] * (1 + wfv["p90"])
    wfv["ci_lo_50"] = wfv["pred_B"] * (1 - wfv["p50"])
    wfv["ci_hi_50"] = wfv["pred_B"] * (1 + wfv["p50"])

    wfv["in_80"] = (wfv["actual_R03"] >= wfv["ci_lo_80"]) & (wfv["actual_R03"] <= wfv["ci_hi_80"])
    wfv["in_50"] = (wfv["actual_R03"] >= wfv["ci_lo_50"]) & (wfv["actual_R03"] <= wfv["ci_hi_50"])

    # Overall coverage
    cov_80 = float(wfv["in_80"].mean())
    cov_50 = float(wfv["in_50"].mean())

    # Coverage by season
    peak_mask = ~wfv["iso_week"].between(OFF_SEASON_START, OFF_SEASON_END)
    off_mask  =  wfv["iso_week"].between(OFF_SEASON_START, OFF_SEASON_END)
    cov_80_peak = float(wfv.loc[peak_mask, "in_80"].mean())
    cov_80_off  = float(wfv.loc[off_mask,  "in_80"].mean())
    cov_50_peak = float(wfv.loc[peak_mask, "in_50"].mean())

    print(f"\n  CI Calibration (nominal vs actual coverage):")
    print(f"    80% CI — actual coverage: {cov_80:.1%}  "
          f"(peak: {cov_80_peak:.1%}, off-season: {cov_80_off:.1%})")
    print(f"    50% CI — actual coverage: {cov_50:.1%}  (peak: {cov_50_peak:.1%})")
    well_80 = abs(cov_80 - 0.80) < 0.10
    well_50 = abs(cov_50 - 0.50) < 0.10
    print(f"    80% CI calibration: {'GOOD' if well_80 else 'OVER-WIDE' if cov_80 > 0.80 else 'UNDER-WIDE'}")
    print(f"    50% CI calibration: {'GOOD' if well_50 else 'OVER-WIDE' if cov_50 > 0.50 else 'UNDER-WIDE'}")

    # Coverage by decile of predicted value
    wfv["pred_decile"] = pd.qcut(wfv["pred_B"], 10, labels=False, duplicates="drop")
    decile_cov = wfv.groupby("pred_decile")["in_80"].mean().reset_index()

    cal = {
        "coverage_80_overall": round(cov_80, 4),
        "coverage_80_peak_season": round(cov_80_peak, 4),
        "coverage_80_off_season":  round(cov_80_off, 4),
        "coverage_50_overall": round(cov_50, 4),
        "coverage_50_peak_season": round(cov_50_peak, 4),
        "nominal_80": 0.80,
        "nominal_50": 0.50,
        "calibration_80": "good" if well_80 else ("over_wide" if cov_80 > 0.80 else "under_wide"),
        "calibration_50": "good" if well_50 else ("over_wide" if cov_50 > 0.50 else "under_wide"),
        "n_observations": len(wfv),
    }
    with open(os.path.join(OUT_DIR, "ci_calibration.json"), "w") as f:
        json.dump(cal, f, indent=2)
    print(f"[OK] Saved: output/ci_calibration.json")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Confidence Interval Calibration — Empirical vs Nominal Coverage\n"
                 "Based on 192 walk-forward validation predictions",
                 fontsize=11, fontweight="bold", color=BLUE)

    # Reliability diagram
    ax = axes[0]
    nominal  = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    actual_cov = []
    for q in nominal:
        err_q = wfv.groupby("iso_week")["abs_rel_err"].quantile(q)
        ci_hi = wfv["pred_B"] * (1 + wfv["iso_week"].map(err_q))
        ci_lo = wfv["pred_B"] * (1 - wfv["iso_week"].map(err_q))
        in_ci = (wfv["actual_R03"] >= ci_lo) & (wfv["actual_R03"] <= ci_hi)
        actual_cov.append(float(in_ci.mean()))

    ax.plot([0,1],[0,1], color=GREY, lw=1.5, linestyle="--", label="Perfect calibration")
    ax.plot(nominal, actual_cov, color=BLUE, lw=2.5, marker="o", ms=7, label="Empirical CI")
    ax.scatter([0.50, 0.80], [cov_50, cov_80], s=120, color=RED, zorder=6,
               label=f"50% CI: {cov_50:.0%}   80% CI: {cov_80:.0%}")
    ax.set_xlabel("Nominal coverage", fontsize=10)
    ax.set_ylabel("Actual coverage", fontsize=10)
    ax.set_title("Reliability diagram", fontsize=10, color=BLUE)
    ax.legend(fontsize=8); ax.grid(alpha=0.25, linestyle="--")
    ax.spines[["top","right"]].set_visible(False)
    ax.set_xlim(0,1); ax.set_ylim(0,1)

    # Coverage by season bar
    ax2 = axes[1]
    cats2 = ["Overall", "Peak season\n(wks 1-21, 40-52)", "Off-season\n(wks 22-39)"]
    vals80 = [cov_80, cov_80_peak, cov_80_off]
    vals50 = [cov_50, cov_50_peak, float(wfv.loc[off_mask, "in_50"].mean())]
    x2 = np.arange(len(cats2)); w2=0.35
    b1 = ax2.bar(x2-w2/2, [v*100 for v in vals80], w2, color=BLUE,  alpha=0.8, label="80% CI", edgecolor="white")
    b2 = ax2.bar(x2+w2/2, [v*100 for v in vals50], w2, color=GREEN, alpha=0.8, label="50% CI", edgecolor="white")
    ax2.bar_label(b1, fmt="%.0f%%", padding=3, fontsize=9)
    ax2.bar_label(b2, fmt="%.0f%%", padding=3, fontsize=9)
    ax2.axhline(80, color=BLUE,  lw=1.2, linestyle=":", alpha=0.6, label="Nominal 80%")
    ax2.axhline(50, color=GREEN, lw=1.2, linestyle=":", alpha=0.6, label="Nominal 50%")
    ax2.set_xticks(x2); ax2.set_xticklabels(cats2, fontsize=9)
    ax2.set_ylabel("Actual coverage (%)", fontsize=9)
    ax2.set_title("Coverage by season", fontsize=10, color=BLUE)
    ax2.legend(fontsize=8); ax2.grid(axis="y", alpha=0.2, linestyle="--")
    ax2.spines[["top","right"]].set_visible(False)
    ax2.set_ylim(0, 115)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "ci_calibration_plot.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {out}")
    return cal


def main():
    print("=" * 65)
    print("STEP 17: SEASONAL SWITCHING RULE + CI CALIBRATION")
    print("=" * 65)

    wfv_df = pd.read_csv(os.path.join(OUT_DIR, "wfv_predictions.csv"))
    df_full = pd.read_csv(
        os.path.join(PROC_DIR, "integrated_dataset.csv"),
        parse_dates=["week_date"], index_col="week_date"
    )

    print("\n--- PART A: SEASONAL SWITCHING RULE ---")
    wfv_sw, sw_metrics = switching_rule(wfv_df, df_full)
    plot_switching(wfv_sw, sw_metrics)

    with open(os.path.join(OUT_DIR, "switching_meta.json"), "w") as f:
        json.dump(sw_metrics, f, indent=2)
    print(f"[OK] Saved: output/switching_meta.json")

    print("\n--- PART B: CI CALIBRATION ---")
    ci_cal = ci_calibration(wfv_df)

    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"  Switching rule improvement : {sw_metrics['improvement_pp']:+.2f} pp MAPE")
    print(f"  80% CI actual coverage     : {ci_cal['coverage_80_overall']:.1%} "
          f"(nominal 80%)")
    print(f"  50% CI actual coverage     : {ci_cal['coverage_50_overall']:.1%} "
          f"(nominal 50%)")
    print("\n[DONE] Switching rule + CI calibration complete.")


if __name__ == "__main__":
    main()
