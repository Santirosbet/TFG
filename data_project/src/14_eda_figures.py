"""
Step 14: EDA Figures for Thesis.

Generates three publication-quality charts that support the EDA and
data description sections of the thesis:

  1. eda_demand_timeseries.png  — R03 & R06 weekly demand (full period)
  2. eda_seasonal_pattern.png   — Mean demand by ISO week with flu season band
  3. eda_ameli_france.png       — AMELI France annual R03/R06 dispensations

Run: python src/14_eda_figures.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "output")

BLUE   = "#1f4e79"
DBLUE  = "#2e75b6"
ORANGE = "#ff6b35"
GREEN  = "#70ad47"
GREY   = "#aaaaaa"
RED    = "#c00000"
LIGHT  = "#d9e8f5"


def plot_demand_timeseries(df):
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle(
        "European Pharmacy Weekly Demand — R03 (Respiratory) and R06 (Antihistamine)\n"
        "Kaggle dataset: single European pharmacy, 2014-2019 (298 weeks)",
        fontsize=12, fontweight="bold", color=BLUE
    )

    for ax, col, color, label in [
        (axes[0], "R03", BLUE,   "R03 — Drugs for Obstructive Airway Diseases"),
        (axes[1], "R06", ORANGE, "R06 — Antihistamines for Systemic Use"),
    ]:
        ax.fill_between(df.index, df[col], alpha=0.18, color=color)
        ax.plot(df.index, df[col], color=color, lw=1.5, label=label)

        mean_val = df[col].mean()
        ax.axhline(mean_val, color=GREY, lw=1, linestyle="--",
                   label=f"Mean: {mean_val:.1f} units/week")

        # shade Jan-Feb (peak flu season) each year
        for year in df.index.year.unique():
            peak_start = pd.Timestamp(f"{year}-01-01")
            peak_end   = pd.Timestamp(f"{year}-02-28")
            ax.axvspan(peak_start, peak_end, alpha=0.08, color=RED)

        ax.set_ylabel("Units dispensed / week", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.2, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

    axes[1].set_xlabel("Date", fontsize=9)
    axes[1].tick_params(axis="x", rotation=30, labelsize=8)

    red_patch = mpatches.Patch(color=RED, alpha=0.25, label="Jan-Feb peak flu season")
    axes[0].legend(
        handles=[axes[0].get_lines()[0], axes[0].get_lines()[1], red_patch],
        fontsize=8, loc="upper right"
    )

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "eda_demand_timeseries.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {out}")


def plot_seasonal_pattern(df):
    df = df.copy()
    df["iso_week"] = df.index.isocalendar().week.astype(int)

    r03_seasonal = df.groupby("iso_week")["R03"].agg(["mean", "std"]).reset_index()
    r06_seasonal = df.groupby("iso_week")["R06"].agg(["mean", "std"]).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Seasonal Demand Pattern by ISO Week\nMean ± 1 SD across all years (2014-2019)",
        fontsize=12, fontweight="bold", color=BLUE
    )

    for ax, seasonal, col, title in [
        (axes[0], r03_seasonal, BLUE,   "R03 — Respiratory"),
        (axes[1], r06_seasonal, ORANGE, "R06 — Antihistamines"),
    ]:
        weeks = seasonal["iso_week"]
        ax.bar(weeks, seasonal["mean"], color=col, alpha=0.7, width=0.8, label="Mean demand")
        ax.fill_between(
            weeks,
            seasonal["mean"] - seasonal["std"],
            seasonal["mean"] + seasonal["std"],
            alpha=0.2, color=col, label="±1 SD"
        )

        # flu season shading
        ax.axvspan(1,  12, alpha=0.07, color=RED,   label="Peak season (weeks 1-12)")
        ax.axvspan(40, 52, alpha=0.07, color=ORANGE, label="Pre-season (weeks 40-52)")
        ax.axvspan(22, 36, alpha=0.07, color=GREEN,  label="Off-season (weeks 22-36)")

        ax.set_title(title, fontsize=11, color=BLUE)
        ax.set_xlabel("ISO Week of Year", fontsize=9)
        ax.set_ylabel("Mean units dispensed / week", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.2, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(1, 52)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "eda_seasonal_pattern.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {out}")


def plot_ameli_france():
    path = os.path.join(PROC_DIR, "france_r03_r06_annual.csv")
    if not os.path.exists(path):
        print("[WARN] france_r03_r06_annual.csv not found — skipping AMELI plot")
        return

    ameli = pd.read_csv(path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "AMELI France — Annual R03 and R06 Dispensations (Open Medic, 2014-2024)\n"
        "Full national French market: ~52 million R03 boxes/year (~1 million/week)",
        fontsize=11, fontweight="bold", color=BLUE
    )

    # Left: stacked bar
    ax = axes[0]
    x = ameli["year"]
    r03 = ameli["france_R03_boxes"] / 1e6
    r06 = ameli["france_R06_boxes"] / 1e6
    ax.bar(x, r03, color=BLUE,   alpha=0.85, label="R03 (respiratory)")
    ax.bar(x, r06, color=ORANGE, alpha=0.85, bottom=r03, label="R06 (antihistamines)")
    ax.set_title("Annual dispensations (stacked)", fontsize=10, color=BLUE)
    ax.set_ylabel("Boxes dispensed (millions)", fontsize=9)
    ax.set_xlabel("Year", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(ax.patches[:len(x)], r03):
        ax.text(bar.get_x() + bar.get_width()/2, val/2,
                f"{val:.0f}M", ha="center", va="center",
                color="white", fontsize=7, fontweight="bold")

    # Right: R03/R06 ratio + YoY growth
    ax2 = axes[1]
    ratio = r03 / r06
    ax2.plot(x, ratio, color=BLUE, lw=2.5, marker="o", ms=6, label="R03/R06 ratio")
    ax2.axhline(ratio.mean(), color=GREY, lw=1, linestyle="--",
                label=f"Mean ratio: {ratio.mean():.2f}x")
    ax2.fill_between(x, ratio, ratio.mean(), where=(ratio > ratio.mean()),
                     alpha=0.15, color=BLUE)
    ax2.fill_between(x, ratio, ratio.mean(), where=(ratio < ratio.mean()),
                     alpha=0.15, color=ORANGE)
    ax2.set_title("R03/R06 demand ratio trend", fontsize=10, color=BLUE)
    ax2.set_ylabel("R03 / R06 ratio", fontsize=9)
    ax2.set_xlabel("Year", fontsize=9)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.25, linestyle="--")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_ylim(0.9, 1.3)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "eda_ameli_france.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {out}")


def main():
    print("=" * 65)
    print("STEP 14: EDA FIGURES")
    print("=" * 65)

    df = pd.read_csv(
        os.path.join(PROC_DIR, "integrated_dataset.csv"),
        parse_dates=["week_date"], index_col="week_date"
    )
    print(f"[OK] Loaded integrated dataset: {len(df)} weeks")

    plot_demand_timeseries(df)
    plot_seasonal_pattern(df)
    plot_ameli_france()
    print("\n[DONE] EDA figures complete.")


if __name__ == "__main__":
    main()
