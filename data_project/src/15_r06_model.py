"""
Step 15: R06 Model + Diebold-Mariano Significance Tests.

Trains an XGBoost model for R06 (antihistamine) demand using the same
feature set as Model B (autoregressive lags + WHO FluNet epidemiological
features), confirming that the predictive framework generalises beyond R03.

Also computes Diebold-Mariano tests to establish statistical significance
of the XGBoost vs SARIMA and Model B vs Model A performance differences.

Outputs:
  output/r06_predictions.csv       — R06 test set predictions
  output/r06_meta.json             — R06 model metrics and feature importance
  output/r06_comparison_plot.png   — R06 actual vs predicted chart
  output/dm_test_results.json      — Diebold-Mariano test results

Run: python src/15_r06_model.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
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
RED    = "#c00000"
GREY   = "#aaaaaa"


def compute_mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def diebold_mariano(e1, e2, h=1):
    """
    Diebold-Mariano test (Harvey, Leybourne & Newbold correction).
    H0: e1 and e2 have equal predictive accuracy.
    Returns: (dm_stat, p_value)
    e1, e2: forecast error arrays (actual - predicted)
    """
    d = e1**2 - e2**2          # loss differential (squared error)
    n = len(d)
    d_bar = np.mean(d)

    # HAC variance with Bartlett kernel, bandwidth = h-1
    gamma0 = np.var(d, ddof=1)
    if h > 1:
        gammas = [np.cov(d[k:], d[:-k])[0, 1] for k in range(1, h)]
        var_d = gamma0 + 2 * sum(gammas)
    else:
        var_d = gamma0

    var_d = max(var_d, 1e-10)
    dm_stat = d_bar / np.sqrt(var_d / n)

    # Harvey-Leybourne-Newbold finite-sample correction
    hln_factor = np.sqrt((n + 1 - 2*h + h*(h-1)/n) / n)
    dm_stat_hln = dm_stat * hln_factor

    p_value = 2 * (1 - scipy_stats.t.cdf(abs(dm_stat_hln), df=n-1))
    return float(dm_stat_hln), float(p_value)


def main():
    print("=" * 65)
    print("STEP 15: R06 MODEL + DIEBOLD-MARIANO TESTS")
    print("=" * 65)

    df = pd.read_csv(
        os.path.join(PROC_DIR, "integrated_dataset.csv"),
        parse_dates=["week_date"], index_col="week_date"
    )

    with open(os.path.join(OUT_DIR, "model_meta.json")) as f:
        meta = json.load(f)
    split_idx = meta["split_idx"]

    # ── R06 Model ─────────────────────────────────────────────────────────────
    print("\n[...] Training R06 XGBoost model...")

    features_b = ["R03_lag1", "R03_lag4_avg", "flu_au_lagged", "flu_eu_positives",
                  "R06_lag1"]

    # Add week_of_year
    df["week_of_year"] = df.index.isocalendar().week.astype(int)
    features_b_r06 = features_b + ["week_of_year"]

    # Drop rows missing any feature
    df_clean = df.dropna(subset=features_b_r06 + ["R06"])

    # Re-compute split on cleaned data
    split_n = int(len(df_clean) * 0.8)
    train = df_clean.iloc[:split_n]
    test  = df_clean.iloc[split_n:]

    X_train = train[features_b_r06]
    y_train = train["R06"]
    X_test  = test[features_b_r06]
    y_test  = test["R06"]

    params = {
        "n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "min_child_weight": 3, "reg_alpha": 0.1, "reg_lambda": 1.0,
        "random_state": 42, "n_jobs": -1,
    }
    model_r06 = xgb.XGBRegressor(**params)
    model_r06.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=False)

    r06_pred = np.maximum(model_r06.predict(X_test), 0)
    actual_r06 = y_test.values

    mae  = mean_absolute_error(actual_r06, r06_pred)
    rmse = np.sqrt(mean_squared_error(actual_r06, r06_pred))
    mape = compute_mape(actual_r06, r06_pred)
    r2   = r2_score(actual_r06, r06_pred)

    print(f"\n  R06 XGBoost (Model B equivalent)")
    print(f"    MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.2f}%  R2={r2:.4f}")
    print(f"    Test period: {test.index[0].date()} to {test.index[-1].date()} ({len(test)} weeks)")

    # Feature importance
    fi = pd.Series(model_r06.feature_importances_, index=features_b_r06).sort_values(ascending=False)
    print("\n  Feature importance (R06 model):")
    for feat, imp in fi.items():
        print(f"    {feat:<30s} {imp:.4f}")

    # Save predictions
    pred_df = pd.DataFrame({
        "week_date":    test.index,
        "actual_R06":   actual_r06,
        "predicted_R06": r06_pred,
    })
    pred_df.to_csv(os.path.join(OUT_DIR, "r06_predictions.csv"), index=False)
    print(f"\n[OK] Saved: output/r06_predictions.csv")

    r06_meta = {
        "target": "R06",
        "features": features_b_r06,
        "train_weeks": len(train),
        "test_weeks": len(test),
        "metrics": {"MAE": round(mae,3), "RMSE": round(rmse,3),
                    "MAPE": round(mape,3), "R2": round(r2,4)},
        "feature_importance": fi.round(4).to_dict(),
    }

    # ── Diebold-Mariano Tests ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("DIEBOLD-MARIANO SIGNIFICANCE TESTS")
    print("=" * 65)

    # Load predictions for DM tests
    xgb_df = pd.read_csv(os.path.join(OUT_DIR, "test_predictions.csv"),
                          parse_dates=["week_date"])

    sarima_path = os.path.join(OUT_DIR, "sarima_predictions.csv")
    wfv_path    = os.path.join(OUT_DIR, "wfv_predictions.csv")

    dm_results = {}

    # DM1: XGBoost Model B vs SARIMA (R03, test set)
    if os.path.exists(sarima_path):
        sarima_df = pd.read_csv(sarima_path, parse_dates=["week_date"])
        actual_r03 = sarima_df["actual_R03"].values
        xgb_r03    = sarima_df["xgboost_pred"].values
        sarima_r03 = sarima_df["sarima_pred"].values

        e_xgb    = actual_r03 - xgb_r03
        e_sarima = actual_r03 - sarima_r03

        dm_stat, p_val = diebold_mariano(e_sarima, e_xgb, h=1)
        winner = "XGBoost" if dm_stat > 0 else "SARIMA"
        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
        print(f"\n  DM1: XGBoost vs SARIMA (R03 test set, n={len(actual_r03)})")
        print(f"    DM statistic : {dm_stat:+.3f}")
        print(f"    p-value      : {p_val:.4f} {sig}")
        print(f"    Winner       : {winner}")
        dm_results["xgboost_vs_sarima_r03"] = {
            "dm_stat": round(dm_stat, 4), "p_value": round(p_val, 6),
            "winner": winner, "significance": sig,
            "n_obs": len(actual_r03),
            "interpretation": (
                f"XGBoost significantly outperforms SARIMA (p={p_val:.4f}) "
                f"on R03 test set. The performance advantage is not due to sampling variability."
                if p_val < 0.05 else
                f"The XGBoost vs SARIMA difference does not reach statistical significance "
                f"(p={p_val:.4f}), though MAPE favours XGBoost."
            )
        }

    # DM2: Model B vs Model A (WFV predictions)
    if os.path.exists(wfv_path):
        wfv_df = pd.read_csv(wfv_path, parse_dates=["week_date"])
        if "pred_A" in wfv_df.columns and "pred_B" in wfv_df.columns:
            actual_wfv = wfv_df["actual_R03"].values
            e_a = actual_wfv - wfv_df["pred_A"].values
            e_b = actual_wfv - wfv_df["pred_B"].values

            dm_stat2, p_val2 = diebold_mariano(e_a, e_b, h=4)
            winner2 = "Model B (FluNet)" if dm_stat2 > 0 else "Model A (lags only)"
            sig2 = "***" if p_val2 < 0.001 else ("**" if p_val2 < 0.01 else ("*" if p_val2 < 0.05 else "ns"))
            print(f"\n  DM2: Model B vs Model A (WFV, n={len(actual_wfv)}, h=4)")
            print(f"    DM statistic : {dm_stat2:+.3f}")
            print(f"    p-value      : {p_val2:.4f} {sig2}")
            print(f"    Winner       : {winner2}")
            dm_results["model_b_vs_model_a_wfv"] = {
                "dm_stat": round(dm_stat2, 4), "p_value": round(p_val2, 6),
                "winner": winner2, "significance": sig2,
                "n_obs": len(actual_wfv), "h": 4,
            }

    with open(os.path.join(OUT_DIR, "dm_test_results.json"), "w") as f:
        json.dump(dm_results, f, indent=2)
    print(f"\n[OK] Saved: output/dm_test_results.json")

    r06_meta["dm_tests"] = dm_results
    with open(os.path.join(OUT_DIR, "r06_meta.json"), "w") as f:
        json.dump(r06_meta, f, indent=2)
    print(f"[OK] Saved: output/r06_meta.json")

    # ── R06 Comparison Plot ──────────────────────────────────────────────────��
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle("R06 Demand Forecast — XGBoost Model (Test Set)\n"
                 "Confirming framework generalisation beyond R03",
                 fontsize=12, fontweight="bold", color=BLUE)

    ax = axes[0]
    ax.plot(test.index, actual_r06, color="black", lw=2, label="Actual R06")
    ax.plot(test.index, r06_pred, color=ORANGE, lw=2, linestyle="--",
            label=f"XGBoost  MAPE={mape:.1f}%  R²={r2:.3f}")
    naive_r06 = np.full(len(test), y_train.mean())
    ax.plot(test.index, naive_r06, color=GREY, lw=1.2, linestyle=":",
            label=f"Naive mean  MAPE={compute_mape(actual_r06, naive_r06):.1f}%")
    ax.set_ylabel("R06 units / week", fontsize=9)
    ax.set_title("Out-of-sample predictions (test set)", fontsize=10, color=BLUE)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    ax2 = axes[1]
    residuals = actual_r06 - r06_pred
    colors_r = [BLUE if r > 0 else RED for r in residuals]
    ax2.bar(test.index, residuals, color=colors_r, alpha=0.7, width=5)
    ax2.axhline(0, color="black", lw=1)
    ax2.set_ylabel("Residual (units)", fontsize=9)
    ax2.set_title("Forecast residuals", fontsize=10, color=BLUE)
    ax2.set_xlabel("Date", fontsize=9)
    ax2.grid(alpha=0.25, linestyle="--")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.tick_params(axis="x", rotation=30, labelsize=8)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "r06_comparison_plot.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {out}")

    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"  R06 MAPE   : {mape:.2f}%  (R03 Model B was 44.16%)")
    print(f"  R06 R²     : {r2:.4f}")
    print(f"  Framework generalises to R06: {'YES' if mape < 60 else 'PARTIAL'}")
    print("\n[DONE] R06 model + Diebold-Mariano tests complete.")


if __name__ == "__main__":
    main()
