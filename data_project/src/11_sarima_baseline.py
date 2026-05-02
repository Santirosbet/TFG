"""
Step 11: SARIMA Baseline Model — comparison with XGBoost.

Fits a seasonal ARIMA model to the R03 training series (univariate,
no exogenous features) and generates a 60-week out-of-sample forecast.
Comparison against XGBoost Model B quantifies the marginal value of
the epidemiological features (WHO FluNet).

Uses pmdarima auto_arima for order selection (stepwise BIC search),
then statsmodels SARIMAX for final fit and diagnostics.

Outputs:
  output/sarima_predictions.csv   — week-by-week SARIMA vs XGBoost vs actual
  output/sarima_meta.json         — model order, metrics, fit info
  output/sarima_diagnostics.png   — residual diagnostics (4-panel)
  output/sarima_forecast_plot.png — forecast comparison chart

Run: python src/11_sarima_baseline.py
"""

import os, json, warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pmdarima as pm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "output")

BLUE   = "#1f4e79"
ORANGE = "#ff6b35"
GREEN  = "#70ad47"
RED    = "#c00000"


def compute_mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def test_stationarity(series, name="series"):
    result = adfuller(series.dropna(), autolag="AIC")
    print(f"  ADF test ({name}): stat={result[0]:.3f}  p={result[1]:.4f}  "
          f"{'STATIONARY' if result[1] < 0.05 else 'NON-STATIONARY'}")
    return result[1] < 0.05


def main():
    print("=" * 65)
    print("STEP 11: SARIMA BASELINE MODEL")
    print("=" * 65)

    df = pd.read_csv(
        os.path.join(PROC_DIR, "integrated_dataset.csv"),
        parse_dates=["week_date"], index_col="week_date"
    )
    df.index.freq = pd.infer_freq(df.index)

    with open(os.path.join(OUT_DIR, "model_meta.json")) as f:
        meta = json.load(f)

    split_idx = meta["split_idx"]
    y_train   = df.iloc[:split_idx]["R03"]
    y_test    = df.iloc[split_idx:]["R03"]
    n_test    = len(y_test)

    print(f"[OK] Train: {len(y_train)} weeks  |  Test: {n_test} weeks")
    print(f"     Train period: {y_train.index[0].date()} -> {y_train.index[-1].date()}")
    print(f"     Test  period: {y_test.index[0].date()  } -> {y_test.index[-1].date()}")

    # ── Stationarity check ────────────────────────────────────────────────────
    print("\n[...] Stationarity tests:")
    test_stationarity(y_train,           "R03 level")
    test_stationarity(y_train.diff(1),   "R03 first diff")
    test_stationarity(y_train.diff(52),  "R03 seasonal diff (52w)")

    # ── Auto ARIMA order selection ─────────────────────────────────────────────
    print("\n[...] Running auto_arima (stepwise, m=52) — this may take ~60 seconds...")
    auto_model = pm.auto_arima(
        y_train,
        m=52,                    # annual seasonal period
        seasonal=True,
        d=1, D=0,                # fix integration orders to speed up search
        max_p=2, max_q=2,
        max_P=1, max_Q=1,
        stepwise=True,
        information_criterion="bic",
        suppress_warnings=True,
        error_action="ignore",
        trace=False,
    )
    order         = auto_model.order
    seasonal_order = auto_model.seasonal_order
    print(f"[OK] Selected order: SARIMA{order}x{seasonal_order}")
    print(f"     AIC={auto_model.aic():.1f}  BIC={auto_model.bic():.1f}")

    # ── Fit SARIMAX for full diagnostics ──────────────────────────────────────
    print("\n[...] Fitting final SARIMAX model...")
    sarima_model = SARIMAX(
        y_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = sarima_model.fit(disp=False, maxiter=200)
    print(f"[OK] Converged | Log-likelihood: {result.llf:.1f}")
    print(f"     AIC={result.aic:.1f}  BIC={result.bic:.1f}")

    # ── Generate 60-week out-of-sample forecast ───────────────────────────────
    forecast_obj = result.get_forecast(steps=n_test)
    sarima_pred  = forecast_obj.predicted_mean.values
    conf_int     = forecast_obj.conf_int(alpha=0.20)   # 80% CI
    ci_lo        = conf_int.iloc[:, 0].values
    ci_hi        = conf_int.iloc[:, 1].values
    sarima_pred  = np.maximum(sarima_pred, 0)           # demand can't be negative

    # ── Load XGBoost predictions for comparison ───────────────────────────────
    xgb_df    = pd.read_csv(os.path.join(OUT_DIR, "test_predictions.csv"),
                             parse_dates=["week_date"])
    xgb_pred  = xgb_df["predicted_R03"].values
    actual    = y_test.values

    # ── Metrics ───────────────────────────────────────────────────────────────
    def metrics(y_true, y_pred, name):
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = compute_mape(y_true, y_pred)
        r2   = r2_score(y_true, y_pred)
        print(f"\n  {name}")
        print(f"    MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.2f}%  R2={r2:.4f}")
        return dict(model=name, MAE=round(mae,3), RMSE=round(rmse,3),
                    MAPE=round(mape,3), R2=round(r2,4))

    print("\n--- Model Comparison (Test Set) ---")
    m_sarima = metrics(actual, sarima_pred, "SARIMA")
    m_xgb    = metrics(actual, xgb_pred,   "XGBoost (Model B)")
    naive_pred = np.full(n_test, y_train.mean())
    m_naive  = metrics(actual, naive_pred, "Naive (mean)")

    # ── Save predictions CSV ───────────────────────────────────────────────────
    pred_df = pd.DataFrame({
        "week_date":       y_test.index,
        "actual_R03":      actual,
        "sarima_pred":     sarima_pred,
        "sarima_ci_lo":    ci_lo,
        "sarima_ci_hi":    ci_hi,
        "xgboost_pred":    xgb_pred,
        "naive_pred":      naive_pred,
    })
    pred_df.to_csv(os.path.join(OUT_DIR, "sarima_predictions.csv"), index=False)
    print(f"\n[OK] Saved: output/sarima_predictions.csv")

    # ── Save metadata JSON ────────────────────────────────────────────────────
    # ── Ljung-Box test on residuals ──────────────────────────────────────────��
    residuals_lb = result.resid.dropna()
    lb_results = acorr_ljungbox(residuals_lb, lags=[10, 20], return_df=True)
    lb_10_stat = float(lb_results["lb_stat"].iloc[0])
    lb_10_pval = float(lb_results["lb_pvalue"].iloc[0])
    lb_20_stat = float(lb_results["lb_stat"].iloc[1])
    lb_20_pval = float(lb_results["lb_pvalue"].iloc[1])

    print(f"\n  Ljung-Box test for residual autocorrelation:")
    print(f"    Q(10) = {lb_10_stat:.3f}  p = {lb_10_pval:.4f}  "
          f"{'[WHITE NOISE]' if lb_10_pval > 0.05 else '[AUTOCORRELATED]'}")
    print(f"    Q(20) = {lb_20_stat:.3f}  p = {lb_20_pval:.4f}  "
          f"{'[WHITE NOISE]' if lb_20_pval > 0.05 else '[AUTOCORRELATED]'}")

    sarima_meta = {
        "order":              list(order),
        "seasonal_order":     list(seasonal_order),
        "aic":                round(result.aic, 2),
        "bic":                round(result.bic, 2),
        "log_likelihood":     round(result.llf, 2),
        "train_weeks":        len(y_train),
        "test_weeks":         n_test,
        "metrics_sarima":     m_sarima,
        "metrics_xgboost":    m_xgb,
        "metrics_naive":      m_naive,
        "sarima_vs_xgb_mape_delta": round(m_sarima["MAPE"] - m_xgb["MAPE"], 3),
        "sarima_vs_naive_mape_delta": round(m_sarima["MAPE"] - m_naive["MAPE"], 3),
        "ljung_box": {
            "Q10_stat":  round(lb_10_stat, 3),
            "Q10_pval":  round(lb_10_pval, 4),
            "Q20_stat":  round(lb_20_stat, 3),
            "Q20_pval":  round(lb_20_pval, 4),
            "residuals_white_noise": bool(lb_10_pval > 0.05),
        },
    }
    with open(os.path.join(OUT_DIR, "sarima_meta.json"), "w") as f:
        json.dump(sarima_meta, f, indent=2)
    print(f"[OK] Saved: output/sarima_meta.json")

    # ── Diagnostic plots ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"SARIMA{order}x{seasonal_order} — Residual Diagnostics",
                 fontsize=13, fontweight="bold", color=BLUE, y=0.98)
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    residuals = result.resid.dropna()

    # Residuals over time
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(residuals.index, residuals.values, color=BLUE, lw=0.8, alpha=0.8)
    ax0.axhline(0, color="#ccc", linewidth=1)
    ax0.fill_between(residuals.index, residuals.values, 0,
                     where=(residuals.values > 0), alpha=0.3, color=ORANGE)
    ax0.fill_between(residuals.index, residuals.values, 0,
                     where=(residuals.values < 0), alpha=0.3, color=RED)
    ax0.set_title("Residuals over time", fontsize=10, color=BLUE)
    ax0.set_ylabel("Residual (units)")
    ax0.spines[["top", "right"]].set_visible(False)
    ax0.grid(alpha=0.2, linestyle="--")

    # Histogram
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.hist(residuals.values, bins=25, color=BLUE, alpha=0.75, edgecolor="white")
    from scipy import stats as scipy_stats
    mu, std = residuals.mean(), residuals.std()
    x_range = np.linspace(residuals.min(), residuals.max(), 100)
    ax1.plot(x_range, scipy_stats.norm.pdf(x_range, mu, std) * len(residuals) * (residuals.max()-residuals.min())/25,
             color=ORANGE, lw=2, label="Normal fit")
    ax1.set_title("Residual distribution", fontsize=10, color=BLUE)
    ax1.set_xlabel("Residual"); ax1.set_ylabel("Frequency")
    ax1.legend(fontsize=8)
    ax1.spines[["top", "right"]].set_visible(False)

    # Q-Q plot
    ax2 = fig.add_subplot(gs[1, 1])
    (osm, osr), (slope, intercept, r) = scipy_stats.probplot(residuals.values, dist="norm")
    ax2.scatter(osm, osr, s=12, color=BLUE, alpha=0.6)
    ax2.plot(osm, slope * np.array(osm) + intercept, color=ORANGE, lw=2)
    ax2.set_title(f"Q-Q Plot (r={r:.3f})", fontsize=10, color=BLUE)
    ax2.set_xlabel("Theoretical quantiles"); ax2.set_ylabel("Sample quantiles")
    ax2.spines[["top", "right"]].set_visible(False)

    fig.savefig(os.path.join(OUT_DIR, "sarima_diagnostics.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: output/sarima_diagnostics.png")

    # ── Forecast comparison plot ──────────────────────────────────────────────
    fig2, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig2.suptitle("SARIMA vs XGBoost — Test Set Forecast Comparison",
                  fontsize=13, fontweight="bold", color=BLUE)

    ax = axes[0]
    ax.plot(y_test.index, actual, color="black", lw=2, label="Actual R03 demand")
    ax.fill_between(y_test.index, ci_lo, ci_hi, alpha=0.15, color=GREEN, label="SARIMA 80% CI")
    ax.plot(y_test.index, sarima_pred, color=GREEN, lw=1.8, linestyle="--",
            label=f"SARIMA{order}x{seasonal_order}  MAPE={m_sarima['MAPE']:.1f}%")
    ax.plot(y_test.index, xgb_pred, color=ORANGE, lw=1.8, linestyle="--",
            label=f"XGBoost (Model B)  MAPE={m_xgb['MAPE']:.1f}%")
    ax.set_ylabel("R03 units / week", fontsize=10)
    ax.set_title("Out-of-sample forecast (60-week test set)", fontsize=11, color=BLUE)
    ax.legend(fontsize=9); ax.grid(alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    # Error comparison bars
    ax2b = axes[1]
    models_cmp   = ["Naive (mean)", f"SARIMA{order}", "XGBoost"]
    mape_vals    = [m_naive["MAPE"], m_sarima["MAPE"], m_xgb["MAPE"]]
    mae_vals     = [m_naive["MAE"],  m_sarima["MAE"],  m_xgb["MAE"]]
    colors_cmp   = ["#aaaaaa", GREEN, ORANGE]
    x = np.arange(len(models_cmp))
    w = 0.35
    b1 = ax2b.bar(x - w/2, mape_vals, w, color=colors_cmp, alpha=0.85, label="MAPE (%)", edgecolor="white")
    ax2b.bar_label(b1, fmt="%.1f%%", padding=3, fontsize=9)
    ax2b.set_ylabel("MAPE (%)", color="#333")
    ax2b_r = ax2b.twinx()
    ax2b_r.plot(x, mae_vals, "o--", color=BLUE, lw=2, ms=8, label="MAE (units)")
    ax2b_r.set_ylabel("MAE (units)", color=BLUE)
    ax2b.set_xticks(x); ax2b.set_xticklabels(models_cmp, fontsize=10)
    ax2b.set_title("Error metric comparison", fontsize=11, color=BLUE)
    ax2b.spines[["top"]].set_visible(False)
    ax2b.grid(axis="y", alpha=0.25, linestyle="--")

    lines1, labs1 = ax2b.get_legend_handles_labels()
    lines2, labs2 = ax2b_r.get_legend_handles_labels()
    ax2b.legend(lines1 + lines2, labs1 + labs2, fontsize=9, loc="upper right")

    plt.tight_layout()
    fig2.savefig(os.path.join(OUT_DIR, "sarima_forecast_plot.png"), dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"[OK] Saved: output/sarima_forecast_plot.png")

    print("\n" + "=" * 65)
    print("SARIMA SUMMARY")
    print("=" * 65)
    delta = sarima_meta["sarima_vs_xgb_mape_delta"]
    winner = "XGBoost" if delta > 0 else "SARIMA"
    print(f"  SARIMA{order}x{seasonal_order}")
    print(f"  SARIMA  MAPE: {m_sarima['MAPE']:.2f}%  MAE: {m_sarima['MAE']:.2f}")
    print(f"  XGBoost MAPE: {m_xgb['MAPE']:.2f}%  MAE: {m_xgb['MAE']:.2f}")
    print(f"  Winner: {winner} (delta MAPE: {abs(delta):.2f} pp)")
    print(f"\n[DONE] SARIMA baseline complete.")


if __name__ == "__main__":
    main()
