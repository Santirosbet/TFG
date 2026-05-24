"""
Step 30 - Backtest Visualization
=================================
Generates publication-quality figures from the enriched model backtest:
  1. Feature importance bar chart
  2. Season-by-season predictions vs actuals (4-panel)
  3. Walk-forward validation time series
  4. Season summary table (for thesis)

Outputs: output/fig_feature_importance.png
         output/fig_backtest_seasons.png
         output/fig_walkforward.png
"""

import os, sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT  = os.path.join(ROOT, "output")

# ── colour palette ──────────────────────────────────────────────────────────
C_ACTUAL  = "#2C3E50"
C_PRED    = "#E74C3C"
C_FILL    = "#FADBD8"
C_GRID    = "#ECF0F1"
SEASON_COLORS = ["#3498DB", "#27AE60", "#E67E22"]

def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor("white")
    ax.grid(True, color=C_GRID, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:  ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    if xlabel: ax.set_xlabel(xlabel, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9)

# ── load data ────────────────────────────────────────────────────────────────
with open(os.path.join(OUT, "enriched_meta.json")) as f:
    meta = json.load(f)

backtest_df = pd.read_csv(os.path.join(OUT, "enriched_backtest.csv"))
pred_df     = pd.read_csv(os.path.join(OUT, "enriched_predictions.csv"),
                          parse_dates=["week_date"])
feats_df    = pd.read_csv(os.path.join(OUT, "enriched_features.csv"),
                          parse_dates=["week_date"])

print("[INFO] Backtest table:")
print(backtest_df.to_string(index=False))

# ── FIGURE 1 : Feature Importance ───────────────────────────────────────────
feat_imp = meta["feature_importances"]
# human-readable names
rename = {
    "week_cos":             "Seasonal cosine (week)",
    "month_cos":            "Seasonal cosine (month)",
    "trends_gripe_es_lag1": "Google Trends: 'gripe' ES (t-1)",
    "trends_flu_eu_lag1":   "Google Trends: 'flu' EU (t-1)",
    "RSV_lag1":             "RSV positives EU (t-1)",
    "temp_eu_lag2":         "EU temperature (t-2)",
    "temp_eu_roll4":        "EU temperature 4-wk avg",
    "RSV_roll4":            "RSV 4-wk rolling avg",
    "R03_lag1":             "R03 demand lag 1w",
    "R03_roll8":            "R03 demand 8-wk avg",
    "flu_au_roll8_lag26":   "AU flu 8-wk avg (t-26w)",
    "flu_au_lag27":         "AU flu positives (t-27w)",
    "R03_roll4":            "R03 demand 4-wk avg",
    "flu_eu_lag1":          "EU flu positives (t-1)",
    "year_num":             "Year trend",
    "R03_lag52":            "R03 demand lag 52w (seasonal)",
    "temp_eu_lag4":         "EU temperature (t-4)",
    "RSV_lag4":             "RSV positives EU (t-4)",
    "flu_au_lag28":         "AU flu positives (t-28w)",
    "RSV_lag2":             "RSV positives EU (t-2)",
}
labels = [rename.get(k, k) for k in feat_imp.keys()]
values = list(feat_imp.values())

# Colour by feature category
cat_colors = []
for k in feat_imp.keys():
    if "week" in k or "month" in k or "year" in k:
        cat_colors.append("#9B59B6")  # purple = seasonal
    elif "trend" in k:
        cat_colors.append("#E67E22")  # orange = Google Trends
    elif "RSV" in k:
        cat_colors.append("#E74C3C")  # red = RSV
    elif "temp" in k or "humid" in k:
        cat_colors.append("#3498DB")  # blue = temperature
    elif "flu_au" in k:
        cat_colors.append("#1ABC9C")  # teal = Australian signal
    elif "flu_eu" in k:
        cat_colors.append("#27AE60")  # green = European flu
    else:
        cat_colors.append("#95A5A6")  # grey = demand autoregression

fig1, ax1 = plt.subplots(figsize=(10, 7))
bars = ax1.barh(range(len(labels)), values[::-1], color=cat_colors[::-1],
                edgecolor="white", linewidth=0.5)
ax1.set_yticks(range(len(labels)))
ax1.set_yticklabels(labels[::-1], fontsize=9)
style_ax(ax1, "Feature Importance — Enriched XGBoost Model",
         xlabel="Importance score (XGBoost gain)", ylabel="")

# legend
legend_els = [
    mpatches.Patch(color="#9B59B6", label="Seasonality"),
    mpatches.Patch(color="#E67E22", label="Google Trends"),
    mpatches.Patch(color="#E74C3C", label="RSV activity"),
    mpatches.Patch(color="#3498DB", label="Temperature"),
    mpatches.Patch(color="#1ABC9C", label="Australian flu signal"),
    mpatches.Patch(color="#27AE60", label="European flu"),
    mpatches.Patch(color="#95A5A6", label="Demand AR"),
]
ax1.legend(handles=legend_els, loc="lower right", fontsize=8, framealpha=0.9)

plt.tight_layout()
fig1.savefig(os.path.join(OUT, "fig_feature_importance.png"), dpi=180, bbox_inches="tight")
plt.close(fig1)
print("[OK] fig_feature_importance.png")

# ── FIGURE 2 : Season-by-season backtest ────────────────────────────────────
seasons_def = [
    {"name": "2016-17", "train_end": "2016-06-01",
     "pred_start": "2016-11-01", "pred_end": "2017-03-31"},
    {"name": "2017-18", "train_end": "2017-06-01",
     "pred_start": "2017-11-01", "pred_end": "2018-03-31"},
    {"name": "2018-19", "train_end": "2018-06-01",
     "pred_start": "2018-11-01", "pred_end": "2019-03-31"},
]

import xgboost as xgb
import warnings; warnings.filterwarnings("ignore")

# rebuild model with best_params and curated features
best_params = meta["best_params"]
feat_cols   = [c for c in meta["features"] if c in feats_df.columns]

fig2, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
fig2.suptitle("Season-by-Season Backtest: Actual vs Predicted R03 Demand",
              fontsize=13, fontweight="bold", y=1.02)

for idx, (s, ax, sc) in enumerate(zip(seasons_def, axes, SEASON_COLORS)):
    # prepare data (must have all features)
    d_clean = feats_df[feat_cols + ["R03", "week_date"]].dropna()
    train_mask = d_clean["week_date"] < s["train_end"]
    test_mask  = (d_clean["week_date"] >= s["pred_start"]) & \
                 (d_clean["week_date"] <= s["pred_end"])

    s_tr = d_clean[train_mask]
    s_te = d_clean[test_mask]

    if len(s_tr) < 52 or s_te.empty:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes)
        continue

    m = xgb.XGBRegressor(**best_params, verbosity=0)
    m.fit(s_tr[feat_cols].values, s_tr["R03"].values, verbose=False)
    preds  = np.maximum(m.predict(s_te[feat_cols].values), 0)
    actual = s_te["R03"].values
    dates  = s_te["week_date"].values

    season_mape = float(np.mean(np.abs((actual - preds) / np.maximum(actual, 1))) * 100)
    total_err   = (preds.sum() - actual.sum()) / max(actual.sum(), 1) * 100

    # Plot
    ax.fill_between(dates, preds, actual,
                    where=preds < actual, alpha=0.25, color="red", label="_under")
    ax.fill_between(dates, preds, actual,
                    where=preds >= actual, alpha=0.2, color="green", label="_over")
    ax.plot(dates, actual, color=C_ACTUAL, linewidth=2.0, label="Actual", zorder=3)
    ax.plot(dates, preds,  color=sc,       linewidth=1.8, linestyle="--",
            label="Predicted", zorder=3)

    # Annotate
    ax.annotate(
        f"MAPE: {season_mape:.1f}%\nTotal error: {total_err:+.1f}%",
        xy=(0.03, 0.96), xycoords="axes fraction",
        fontsize=8.5, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85)
    )
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    style_ax(ax, f"Season {s['name']}", ylabel="Units / week" if idx == 0 else "")
    ax.legend(fontsize=7, loc="upper right")

plt.tight_layout()
fig2.savefig(os.path.join(OUT, "fig_backtest_seasons.png"), dpi=180, bbox_inches="tight")
plt.close(fig2)
print("[OK] fig_backtest_seasons.png")

# ── FIGURE 3 : Walk-forward validation ──────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(13, 5))

ax3.plot(pred_df["week_date"], pred_df["R03"],
         color=C_ACTUAL, linewidth=2.0, label="Actual R03 demand")
ax3.plot(pred_df["week_date"], pred_df["predicted"],
         color=C_PRED, linewidth=1.6, linestyle="--", label="Model prediction")
ax3.fill_between(pred_df["week_date"], pred_df["R03"], pred_df["predicted"],
                 alpha=0.18, color=C_PRED)

# Shade the severe season
severe_start = pd.Timestamp("2018-11-01")
severe_end   = pd.Timestamp("2019-03-31")
ax3.axvspan(severe_start, severe_end, alpha=0.1, color="orange",
            label="Season 2018-19 (severe)")

wfv_mape = meta["wfv_mean_mape_pct"]
ax3.annotate(
    f"Walk-forward validation MAPE: {wfv_mape:.1f}%",
    xy=(0.02, 0.95), xycoords="axes fraction", fontsize=9, va="top",
    bbox=dict(boxstyle="round", fc="white", alpha=0.9)
)
style_ax(ax3, "Test Set: Enriched XGBoost Predictions vs Actual Demand (R03)",
         xlabel="Week", ylabel="Units / week")
ax3.legend(fontsize=9)
plt.tight_layout()
fig3.savefig(os.path.join(OUT, "fig_walkforward.png"), dpi=180, bbox_inches="tight")
plt.close(fig3)
print("[OK] fig_walkforward.png")

# ── Summary stats ────────────────────────────────────────────────────────────
avg_mape = backtest_df["MAPE temporada (%)"].mean()
avg_total_err = backtest_df["Error total (%)"].abs().mean()
print(f"\n=== SUMMARY FOR THESIS ===")
print(f"  Enriched model avg season MAPE : {avg_mape:.1f}%")
print(f"  Avg absolute total demand error: {avg_total_err:.1f}%")
print(f"  Top external signal             : Google Trends 'gripe' (ES, t-1)")
print(f"  Australian lead-lag importance  : {sum(v for k,v in feat_imp.items() if 'flu_au' in k)*100:.1f}% combined")
print(f"  RSV importance                  : {sum(v for k,v in feat_imp.items() if 'RSV' in k)*100:.1f}% combined")
print(f"  Temperature importance          : {sum(v for k,v in feat_imp.items() if 'temp' in k)*100:.1f}% combined")
print(f"\n[DONE]")
