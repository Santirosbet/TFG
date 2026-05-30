"""
09b: Regenerate shap_bar.png from saved shap_values.csv (no model needed).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")

FEATURE_LABELS = {
    "R03_lag1":         "R03 demand\n(prev. week)",
    "R03_lag4_avg":     "R03 demand\n(4-week avg)",
    "flu_au_positives": "Australia flu\n(current week)",
    "flu_au_lagged":    "Australia flu\n(lagged 26 wk)",
    "flu_eu_positives": "Europe flu\n(current week)",
    "R06_lag1":         "R06 demand\n(prev. week)",
}
BLUE   = "#1f4e79"
ORANGE = "#ff6b35"
GREEN  = "#70ad47"


def plot_bar(shap_vals, feature_names, out_path):
    """Bar chart: mean absolute SHAP value per feature."""
    mean_abs = np.abs(shap_vals).mean(axis=0)
    order = np.argsort(mean_abs)
    labels = [FEATURE_LABELS.get(feature_names[i], feature_names[i]) for i in order]
    values = mean_abs[order]
    colors = [ORANGE if "flu_au" in feature_names[i] else
              GREEN  if "flu_eu" in feature_names[i] else
              BLUE   for i in order]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.55)
    ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=11, color="#444")
    ax.set_xlabel("Mean |SHAP value| — average impact on prediction (units)", fontsize=12)
    ax.set_title("Feature Importance via SHAP — Mean Absolute Impact",
                 fontsize=14, fontweight="bold", color=BLUE, pad=10)
    ax.set_xlim(0, values.max() * 1.28)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.tick_params(labelsize=11)

    legend_patches = [
        mpatches.Patch(color=ORANGE, label="Australian flu signal (lead indicator)"),
        mpatches.Patch(color=GREEN,  label="European flu signal (concurrent)"),
        mpatches.Patch(color=BLUE,   label="Autoregressive demand features"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=11, framealpha=0.85)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] Saved bar chart: {out_path}")


if __name__ == "__main__":
    csv_path = os.path.join(OUT_DIR, "shap_values.csv")
    df = pd.read_csv(csv_path, index_col=0)
    shap_cols = [c for c in df.columns if c.startswith("shap_")]
    feature_names = [c[5:] for c in shap_cols]   # strip "shap_" prefix
    shap_vals = df[shap_cols].values
    plot_bar(shap_vals, feature_names, os.path.join(OUT_DIR, "shap_bar.png"))
    print("Done.")
