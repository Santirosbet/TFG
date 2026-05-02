"""
Step 16: Multi-Country Southern Hemisphere Ensemble Model.

Combines lagged flu signals from Australia (lag 26w), New Zealand (lag 27w),
and Chile (lag 35w) as joint features in a single XGBoost model, testing
whether a multi-signal ensemble reduces MAPE versus the single-country Model B.

Hypothesis: when one country has an anomalous flu season, the others compensate,
producing a more stable epidemiological leading indicator.

Outputs:
  output/ensemble_predictions.csv  — test set predictions for all model variants
  output/ensemble_meta.json        — metrics comparison table
  output/ensemble_plot.png         — comparison chart

Run: python src/16_ensemble_model.py
"""

import os, json, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "output")

BLUE   = "#1f4e79"
ORANGE = "#ff6b35"
GREEN  = "#70ad47"
PURPLE = "#7030a0"
RED    = "#c00000"
GREY   = "#aaaaaa"

# Optimal lags from Section 6.11 (CCF analysis)
COUNTRY_LAGS = {
    "australia":    26,
    "new_zealand":  27,
    "chile":        35,
}

XGB_PARAMS = {
    "n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "min_child_weight": 3, "reg_alpha": 0.1, "reg_lambda": 1.0,
    "random_state": 42, "n_jobs": -1,
}


def compute_mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def load_country_signal(country, lag_weeks):
    path = os.path.join(PROC_DIR, f"flunet_{country}.csv")
    df = pd.read_csv(path, parse_dates=["iso_date"]).set_index("iso_date")
    df = df[["INF_ALL"]].rename(columns={"INF_ALL": f"flu_{country}"})
    df = df.resample("W-SUN").sum()
    df[f"flu_{country}_lag{lag_weeks}"] = df[f"flu_{country}"].shift(lag_weeks)
    return df[[f"flu_{country}_lag{lag_weeks}"]]


def build_dataset():
    base = pd.read_csv(
        os.path.join(PROC_DIR, "integrated_dataset.csv"),
        parse_dates=["week_date"], index_col="week_date"
    )

    for country, lag in COUNTRY_LAGS.items():
        sig = load_country_signal(country, lag)
        base = base.join(sig, how="left")
        print(f"  [OK] {country} lag-{lag}w signal joined "
              f"({sig.dropna().shape[0]} non-null weeks)")

    return base


def train_evaluate(df, features, label, split_idx):
    df_clean = df.dropna(subset=features + ["R03"])
    train = df_clean.iloc[:split_idx]
    test  = df_clean.iloc[split_idx:]

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(train[features], train["R03"],
              eval_set=[(test[features], test["R03"])], verbose=False)

    pred = np.maximum(model.predict(test[features]), 0)
    actual = test["R03"].values

    mape = compute_mape(actual, pred)
    mae  = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    r2   = r2_score(actual, pred)

    print(f"  {label:<40s}  MAPE={mape:.2f}%  MAE={mae:.2f}  R2={r2:.4f}")
    return pred, actual, test.index, {
        "label": label, "MAPE": round(mape,3), "MAE": round(mae,3),
        "RMSE": round(rmse,3), "R2": round(r2,4),
        "n_features": len(features), "test_weeks": len(actual),
    }


def main():
    print("=" * 65)
    print("STEP 16: MULTI-COUNTRY ENSEMBLE MODEL")
    print("=" * 65)

    df = build_dataset()

    with open(os.path.join(OUT_DIR, "model_meta.json")) as f:
        meta = json.load(f)
    split_idx = meta["split_idx"]

    df["week_of_year"] = df.index.isocalendar().week.astype(int)

    base_features = ["R03_lag1", "R03_lag4_avg", "flu_eu_positives", "week_of_year"]

    # Model definitions
    models = [
        ("Model B (Australia only)",
         base_features + ["flu_au_lagged"]),
        ("Ensemble AU+NZ",
         base_features + ["flu_au_lagged",
                          f"flu_new_zealand_lag{COUNTRY_LAGS['new_zealand']}"]),
        ("Ensemble AU+CL",
         base_features + ["flu_au_lagged",
                          f"flu_chile_lag{COUNTRY_LAGS['chile']}"]),
        ("Ensemble AU+NZ+CL",
         base_features + ["flu_au_lagged",
                          f"flu_new_zealand_lag{COUNTRY_LAGS['new_zealand']}",
                          f"flu_chile_lag{COUNTRY_LAGS['chile']}"]),
    ]

    print(f"\n{'Model':<40s}  MAPE       MAE    R2")
    print("-" * 65)

    results = {}
    pred_df_rows = {}
    actual_arr = None
    dates_arr  = None

    for label, features in models:
        pred, actual, dates, metrics = train_evaluate(df, features, label, split_idx)
        results[label] = metrics
        pred_df_rows[label] = pred
        if actual_arr is None:
            actual_arr = actual
            dates_arr  = dates

    # ── Save predictions CSV ─────────────────────────────────────────────────
    pred_df = pd.DataFrame({"week_date": dates_arr, "actual_R03": actual_arr})
    for label, pred in pred_df_rows.items():
        key = label.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "")
        pred_df[key] = pred
    pred_df.to_csv(os.path.join(OUT_DIR, "ensemble_predictions.csv"), index=False)
    print(f"\n[OK] Saved: output/ensemble_predictions.csv")

    # ── Winner analysis ───────────────────────────────────────────────────────
    best_label = min(results, key=lambda k: results[k]["MAPE"])
    model_b_mape = results["Model B (Australia only)"]["MAPE"]
    best_mape    = results[best_label]["MAPE"]
    improvement  = model_b_mape - best_mape

    print(f"\n  Best model  : {best_label}")
    print(f"  MAPE improvement vs Model B: {improvement:+.2f} pp")

    ensemble_meta = {
        "models": results,
        "best_label": best_label,
        "best_mape": best_mape,
        "model_b_mape": model_b_mape,
        "improvement_pp": round(improvement, 3),
        "country_lags": COUNTRY_LAGS,
    }
    with open(os.path.join(OUT_DIR, "ensemble_meta.json"), "w") as f:
        json.dump(ensemble_meta, f, indent=2)
    print(f"[OK] Saved: output/ensemble_meta.json")

    # ── Plot ─────────────────────────────────────────────────────────────────
    colors = [GREY, BLUE, GREEN, PURPLE]
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle("Multi-Country SH Ensemble — Test Set Comparison\n"
                 "Australia only vs AU+NZ vs AU+Chile vs AU+NZ+Chile",
                 fontsize=12, fontweight="bold", color=BLUE)

    ax = axes[0]
    ax.plot(dates_arr, actual_arr, color="black", lw=2, label="Actual R03", zorder=5)
    for (label, _), color, pred in zip(models, colors, pred_df_rows.values()):
        mape_v = results[label]["MAPE"]
        lw = 2.5 if label == best_label else 1.3
        ls = "-" if label == best_label else "--"
        ax.plot(dates_arr, pred, color=color, lw=lw, linestyle=ls,
                label=f"{label}  MAPE={mape_v:.1f}%",
                alpha=1.0 if label == best_label else 0.65)
    ax.set_ylabel("R03 units / week", fontsize=9)
    ax.set_title("Out-of-sample predictions (60-week test set)", fontsize=10, color=BLUE)
    ax.legend(fontsize=8); ax.grid(alpha=0.2, linestyle="--")
    ax.spines[["top","right"]].set_visible(False)

    ax2 = axes[1]
    labels_short = ["AU only", "AU+NZ", "AU+CL", "AU+NZ+CL"]
    mapes  = [results[l]["MAPE"] for l, _ in models]
    bars   = ax2.bar(labels_short, mapes, color=colors, alpha=0.85, edgecolor="white", width=0.55)
    ax2.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=10)
    ax2.axhline(model_b_mape, color=GREY, lw=1.5, linestyle="--",
                label=f"Model B baseline ({model_b_mape:.1f}%)")
    ax2.set_ylabel("MAPE (%)", fontsize=9)
    ax2.set_title("MAPE comparison — single country vs ensemble", fontsize=10, color=BLUE)
    ax2.legend(fontsize=9); ax2.grid(axis="y", alpha=0.2, linestyle="--")
    ax2.spines[["top","right"]].set_visible(False)
    ax2.set_ylim(0, max(mapes) * 1.2)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "ensemble_plot.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {out}")
    print("\n[DONE] Ensemble model complete.")


if __name__ == "__main__":
    main()
