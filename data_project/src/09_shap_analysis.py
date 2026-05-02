"""
Step 9: SHAP Explainability Analysis for the XGBoost demand model.

Generates four publication-quality visualisations:
  1. Beeswarm summary  — global importance + direction for every prediction
  2. Bar summary       — mean |SHAP| ranking (clean feature importance)
  3. Waterfall         — single-prediction decomposition (peak demand week)
  4. Dependence plots  — how flu_au_lagged and R03_lag1 drive the output

Outputs saved to output/shap_*.png + output/shap_values.csv

Run: python src/09_shap_analysis.py
"""

import os, json
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "output")

FEATURE_LABELS = {
    "R03_lag1":         "R03 demand\n(prev. week)",
    "R03_lag4_avg":     "R03 demand\n(4-week avg)",
    "flu_au_positives": "Australia flu\n(current week)",
    "flu_au_lagged":    "Australia flu\n(lagged 26 wk)",
    "flu_eu_positives": "Europe flu\n(current week)",
    "R06_lag1":         "R06 demand\n(prev. week)",
}
BLUE  = "#1f4e79"
ORANGE = "#ff6b35"
GREEN  = "#70ad47"


def load_data():
    df = pd.read_csv(os.path.join(PROC_DIR, "integrated_dataset.csv"),
                     parse_dates=["week_date"], index_col="week_date")
    with open(os.path.join(OUT_DIR, "model_meta.json")) as f:
        meta = json.load(f)
    model = xgb.XGBRegressor()
    model.load_model(os.path.join(OUT_DIR, "xgboost_best_model.json"))
    return df, model, meta["features"]


def plot_beeswarm(shap_vals, X_display, out_path):
    """Beeswarm: one dot per prediction per feature, coloured by feature value."""
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.sca(ax)
    shap.summary_plot(shap_vals, X_display, show=False,
                      color_bar_label="Feature value\n(normalised)",
                      plot_size=None)
    ax = plt.gca()
    ax.set_xlabel("SHAP value — impact on predicted R03 demand (units)", fontsize=11)
    ax.set_title("Global Feature Importance — SHAP Beeswarm\n"
                 "Each dot = one weekly prediction. Colour = feature magnitude.",
                 fontsize=12, fontweight="bold", color=BLUE, pad=12)
    ax.axvline(0, color="#ccc", linewidth=0.8, zorder=0)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved beeswarm: {out_path}")


def plot_bar(shap_vals, feature_names, out_path):
    """Bar chart: mean absolute SHAP value per feature."""
    mean_abs = np.abs(shap_vals).mean(axis=0)
    order = np.argsort(mean_abs)
    labels = [FEATURE_LABELS.get(feature_names[i], feature_names[i]) for i in order]
    values = mean_abs[order]
    colors = [ORANGE if "flu_au" in feature_names[i] else
              GREEN  if "flu_eu" in feature_names[i] else
              BLUE   for i in order]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.55)
    ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=9, color="#444")
    ax.set_xlabel("Mean |SHAP value| — average impact on prediction (units)", fontsize=10)
    ax.set_title("Feature Importance via SHAP — Mean Absolute Impact",
                 fontsize=12, fontweight="bold", color=BLUE, pad=10)
    ax.set_xlim(0, values.max() * 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    legend_patches = [
        mpatches.Patch(color=ORANGE, label="Australian flu signal (lead indicator)"),
        mpatches.Patch(color=GREEN,  label="European flu signal (concurrent)"),
        mpatches.Patch(color=BLUE,   label="Autoregressive demand features"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9, framealpha=0.8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved bar chart: {out_path}")


def plot_waterfall(explainer, shap_explanation, X_test, y_test, out_path):
    """Waterfall for the peak-demand week in the test set."""
    peak_idx = int(y_test.values.argmax())
    peak_date = y_test.index[peak_idx].strftime("%d %b %Y")
    actual    = float(y_test.iloc[peak_idx])

    sv = shap_explanation[peak_idx]

    fig, ax = plt.subplots(figsize=(9, 6))
    plt.sca(ax)
    shap.waterfall_plot(sv, show=False, max_display=10)
    ax = plt.gca()
    ax.set_title(
        f"SHAP Waterfall — Peak Demand Week ({peak_date})\n"
        f"Actual demand: {actual:.1f} units  |  Baseline: {explainer.expected_value:.1f} units",
        fontsize=11, fontweight="bold", color=BLUE, pad=10
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved waterfall ({peak_date}): {out_path}")


def plot_dependence(shap_vals, X_test, feature, interaction, out_path):
    """Dependence plot: SHAP value of 'feature' coloured by 'interaction'."""
    feat_idx = list(X_test.columns).index(feature)
    inter_idx = list(X_test.columns).index(interaction)
    x_vals = X_test.iloc[:, feat_idx].values
    s_vals = shap_vals[:, feat_idx]
    c_vals = X_test.iloc[:, inter_idx].values

    fig, ax = plt.subplots(figsize=(9, 5))
    sc = ax.scatter(x_vals, s_vals, c=c_vals, cmap="RdYlGn_r",
                    alpha=0.75, s=40, edgecolors="none")
    cb = plt.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(FEATURE_LABELS.get(interaction, interaction), fontsize=9)
    ax.axhline(0, color="#ccc", linewidth=0.8)
    ax.set_xlabel(FEATURE_LABELS.get(feature, feature), fontsize=10)
    ax.set_ylabel(f"SHAP value for\n{FEATURE_LABELS.get(feature, feature)}", fontsize=10)
    ax.set_title(
        f"SHAP Dependence — {FEATURE_LABELS.get(feature, feature)}\n"
        f"(coloured by {FEATURE_LABELS.get(interaction, interaction)})",
        fontsize=12, fontweight="bold", color=BLUE, pad=10
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, linestyle="--")
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved dependence ({feature}): {out_path}")


def main():
    print("=" * 60)
    print("STEP 9: SHAP EXPLAINABILITY ANALYSIS")
    print("=" * 60)

    df, model, features = load_data()

    # Chronological 80/20 split — same as model training
    split = int(len(df) * 0.8)
    X_train = df.iloc[:split][features]
    X_test  = df.iloc[split:][features]
    y_test  = df.iloc[split:]["R03"]

    print(f"[OK] Loaded {len(df)} rows | features: {features}")
    print(f"[OK] Test set: {len(X_test)} weeks")

    # ── Compute SHAP values (TreeExplainer is exact for XGBoost) ──────────────
    print("\n[...] Computing SHAP values (TreeExplainer)...")
    explainer    = shap.TreeExplainer(model)
    shap_vals    = explainer.shap_values(X_test)          # (n_samples, n_features)
    explanation  = explainer(X_test)                       # Explanation object for waterfall

    print(f"[OK] SHAP values shape: {shap_vals.shape}")
    print(f"[OK] Baseline (expected value): {explainer.expected_value:.2f} units")

    # ── Print text summary ────────────────────────────────────────────────────
    mean_abs = np.abs(shap_vals).mean(axis=0)
    print("\n--- SHAP Feature Importance (mean |SHAP|) ---")
    for feat, imp in sorted(zip(features, mean_abs), key=lambda x: -x[1]):
        label = FEATURE_LABELS.get(feat, feat).replace("\n", " ")
        bar   = "#" * int(imp / mean_abs.max() * 20)
        print(f"  {label:<35} {imp:6.3f}  {bar}")

    # R-squared from SHAP perspective
    y_pred_shap = explainer.expected_value + shap_vals.sum(axis=1)
    ss_res = ((y_test.values - y_pred_shap) ** 2).sum()
    ss_tot = ((y_test.values - y_test.values.mean()) ** 2).sum()
    print(f"\n[OK] SHAP decomposition R²: {1 - ss_res/ss_tot:.4f} (should match model R²)")

    # ── Display feature for beeswarm (rename columns) ─────────────────────────
    X_display = X_test.rename(columns={k: FEATURE_LABELS.get(k, k).replace("\n", " ")
                                        for k in X_test.columns})

    # ── Save all plots ─────────────────────────────────────────────────────────
    plot_beeswarm(shap_vals, X_display,
                  os.path.join(OUT_DIR, "shap_beeswarm.png"))

    plot_bar(shap_vals, features,
             os.path.join(OUT_DIR, "shap_bar.png"))

    plot_waterfall(explainer, explanation, X_test, y_test,
                   os.path.join(OUT_DIR, "shap_waterfall.png"))

    if "flu_au_lagged" in features and "R03_lag1" in features:
        plot_dependence(shap_vals, X_test,
                        feature="flu_au_lagged", interaction="R03_lag1",
                        out_path=os.path.join(OUT_DIR, "shap_dependence_flu_lag.png"))

    if "R03_lag1" in features:
        plot_dependence(shap_vals, X_test,
                        feature="R03_lag1", interaction="flu_au_lagged",
                        out_path=os.path.join(OUT_DIR, "shap_dependence_r03_lag.png"))

    # ── Save raw SHAP values as CSV ───────────────────────────────────────────
    shap_df = pd.DataFrame(shap_vals, columns=[f"shap_{f}" for f in features],
                            index=X_test.index)
    shap_df["actual_R03"]    = y_test.values
    shap_df["predicted_R03"] = y_pred_shap
    shap_df["baseline"]      = explainer.expected_value
    shap_df.to_csv(os.path.join(OUT_DIR, "shap_values.csv"))
    print(f"[OK] Saved SHAP values CSV: output/shap_values.csv")

    # ── Key findings summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("KEY FINDINGS FOR THESIS")
    print("=" * 60)
    top_feat  = features[int(mean_abs.argmax())]
    top_imp   = mean_abs.max()
    flu_imp   = mean_abs[features.index("flu_au_lagged")] if "flu_au_lagged" in features else 0
    lag1_imp  = mean_abs[features.index("R03_lag1")] if "R03_lag1" in features else 0
    flu_share = flu_imp / mean_abs.sum() * 100

    print(f"  Most important feature : {top_feat} ({top_imp:.3f} avg units impact)")
    print(f"  flu_au_lagged SHAP     : {flu_imp:.3f} units avg impact ({flu_share:.1f}% of total)")
    print(f"  R03_lag1 SHAP          : {lag1_imp:.3f} units avg impact")
    print(f"  Baseline expected value: {explainer.expected_value:.2f} units/week")

    print("\n[DONE] SHAP analysis complete. Outputs in output/shap_*.png")


if __name__ == "__main__":
    main()
