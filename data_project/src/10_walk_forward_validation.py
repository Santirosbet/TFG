"""
Step 10: Walk-Forward Validation for XGBoost demand model.

Uses an expanding-window time-series cross-validation to produce
unbiased out-of-sample performance estimates:

  - Minimum training window : 104 weeks (2 full seasonal cycles)
  - Step size               : 4 weeks  (monthly fold advancement)
  - Test window per fold     : 4 weeks  (each fold predicts 4 weeks ahead)
  - Models compared          : Model A (lag features only)
                               Model B (lags + WHO FluNet)

Outputs:
  output/wfv_fold_results.csv   — per-fold metrics for both models
  output/wfv_summary.csv        — aggregate mean/std/min/max per model
  output/wfv_predictions.csv    — all out-of-sample predictions (for dashboard)
  output/wfv_plot.png           — 3-panel diagnostic plot

Run: python src/10_walk_forward_validation.py
"""

import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "output")
os.makedirs(OUT_DIR, exist_ok=True)

MIN_TRAIN   = 104   # weeks (2 years minimum training window)
STEP        = 4     # weeks per fold advancement
TEST_WINDOW = 4     # weeks predicted per fold
BLUE        = "#1f4e79"
ORANGE      = "#ff6b35"
GREEN       = "#70ad47"
GREY        = "#aaaaaa"


def compute_mape(y_true, y_pred):
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def build_model():
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
    )


def run_walk_forward(df, features_a, features_b):
    """
    Expanding-window walk-forward validation.

    Returns
    -------
    fold_records : list of dicts — one per fold with metrics for A and B
    all_preds    : DataFrame     — full out-of-sample predictions with dates
    """
    n = len(df)
    target = "R03"
    fold_records = []
    all_preds = []

    fold_idx = 0
    train_end = MIN_TRAIN  # exclusive upper bound (iloc)

    while train_end + TEST_WINDOW <= n:
        test_start = train_end
        test_end   = train_end + TEST_WINDOW

        X_tr_a = df.iloc[:train_end][features_a]
        X_tr_b = df.iloc[:train_end][features_b]
        y_tr   = df.iloc[:train_end][target]

        X_te_a = df.iloc[test_start:test_end][features_a]
        X_te_b = df.iloc[test_start:test_end][features_b]
        y_te   = df.iloc[test_start:test_end][target]
        dates  = df.index[test_start:test_end]

        # Model A (lags only)
        m_a = build_model()
        m_a.fit(X_tr_a, y_tr)
        p_a = m_a.predict(X_te_a)

        # Model B (lags + FluNet)
        m_b = build_model()
        m_b.fit(X_tr_b, y_tr)
        p_b = m_b.predict(X_te_b)

        y_arr = y_te.values

        fold_records.append({
            "fold":          fold_idx + 1,
            "train_weeks":   train_end,
            "test_start":    dates[0].strftime("%Y-%m-%d"),
            "test_end":      dates[-1].strftime("%Y-%m-%d"),
            "A_MAE":         mean_absolute_error(y_arr, p_a),
            "A_RMSE":        np.sqrt(mean_squared_error(y_arr, p_a)),
            "A_MAPE":        compute_mape(y_arr, p_a),
            "A_R2":          r2_score(y_arr, p_a),
            "B_MAE":         mean_absolute_error(y_arr, p_b),
            "B_RMSE":        np.sqrt(mean_squared_error(y_arr, p_b)),
            "B_MAPE":        compute_mape(y_arr, p_b),
            "B_R2":          r2_score(y_arr, p_b),
        })

        for i, dt in enumerate(dates):
            all_preds.append({
                "week_date":   dt,
                "actual_R03":  float(y_arr[i]),
                "pred_A":      float(p_a[i]),
                "pred_B":      float(p_b[i]),
                "fold":        fold_idx + 1,
                "train_weeks": train_end,
            })

        fold_idx  += 1
        train_end += STEP

    fold_df = pd.DataFrame(fold_records)
    pred_df = pd.DataFrame(all_preds).set_index("week_date")
    return fold_df, pred_df


def print_summary(fold_df):
    print("\n" + "=" * 70)
    print("WALK-FORWARD VALIDATION — AGGREGATE SUMMARY")
    print("=" * 70)
    for model, prefix in [("Model A  (Lags only)", "A"), ("Model B  (+ FluNet)", "B")]:
        mapes = fold_df[f"{prefix}_MAPE"].dropna()
        maes  = fold_df[f"{prefix}_MAE"]
        rmses = fold_df[f"{prefix}_RMSE"]
        r2s   = fold_df[f"{prefix}_R2"]
        print(f"\n  {model}")
        print(f"    MAPE : {mapes.mean():.2f}%  (std {mapes.std():.2f}%,  "
              f"min {mapes.min():.2f}%,  max {mapes.max():.2f}%)")
        print(f"    MAE  : {maes.mean():.2f}   (std {maes.std():.2f})")
        print(f"    RMSE : {rmses.mean():.2f}   (std {rmses.std():.2f})")
        print(f"    R2   : {r2s.mean():.4f}  (std {r2s.std():.4f})")

    delta_mape = (fold_df["A_MAPE"] - fold_df["B_MAPE"]).mean()
    pct_folds_b_wins = (fold_df["B_MAPE"] < fold_df["A_MAPE"]).mean() * 100
    print(f"\n  FluNet improvement : {delta_mape:+.2f}% avg MAPE reduction")
    print(f"  Model B wins       : {pct_folds_b_wins:.0f}% of folds")
    print("=" * 70)


def plot_wfv(fold_df, pred_df, out_path):
    fig, axes = plt.subplots(3, 1, figsize=(14, 13))
    fig.suptitle(
        "Walk-Forward Validation — XGBoost Demand Model\n"
        f"Expanding window | min_train={MIN_TRAIN}w | step={STEP}w | "
        f"test_window={TEST_WINDOW}w | {len(fold_df)} folds",
        fontsize=13, fontweight="bold", color=BLUE, y=0.98
    )

    # ── Panel 1: Out-of-sample reconstruction ────────────────────────────────
    ax1 = axes[0]
    ax1.plot(pred_df.index, pred_df["actual_R03"],
             color="black", lw=1.5, label="Actual R03", zorder=3)
    ax1.plot(pred_df.index, pred_df["pred_A"],
             color=GREY, lw=1, alpha=0.7, label="Model A (lags)", linestyle="--")
    ax1.plot(pred_df.index, pred_df["pred_B"],
             color=ORANGE, lw=1.5, alpha=0.85, label="Model B (+ FluNet)", linestyle="--")

    # shade fold boundaries
    fold_starts = pred_df.groupby("fold").apply(lambda g: g.index.min())
    for fs in fold_starts[::4]:  # every 4th fold to avoid clutter
        ax1.axvline(fs, color="#e0e0e0", linewidth=0.6, zorder=0)

    ax1.set_ylabel("R03 units / week", fontsize=10)
    ax1.set_title("Out-of-Sample Predictions Across All Folds", fontsize=11, color=BLUE)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.25, linestyle="--")
    ax1.spines[["top", "right"]].set_visible(False)

    # ── Panel 2: Per-fold MAPE ────────────────────────────────────────────────
    ax2 = axes[1]
    folds = fold_df["fold"]
    ax2.plot(folds, fold_df["A_MAPE"], color=GREY, lw=1.2, marker="o", ms=3,
             label="Model A (lags)", alpha=0.8)
    ax2.plot(folds, fold_df["B_MAPE"], color=ORANGE, lw=1.5, marker="o", ms=3,
             label="Model B (+ FluNet)")
    ax2.axhline(fold_df["A_MAPE"].mean(), color=GREY, linestyle=":", lw=1.2)
    ax2.axhline(fold_df["B_MAPE"].mean(), color=ORANGE, linestyle=":", lw=1.2)
    ax2.fill_between(folds, fold_df["A_MAPE"], fold_df["B_MAPE"],
                     where=(fold_df["B_MAPE"] < fold_df["A_MAPE"]),
                     alpha=0.15, color=GREEN, label="B better than A")
    ax2.fill_between(folds, fold_df["A_MAPE"], fold_df["B_MAPE"],
                     where=(fold_df["B_MAPE"] >= fold_df["A_MAPE"]),
                     alpha=0.12, color="red", label="A better than B")
    ax2.set_xlabel("Fold", fontsize=10)
    ax2.set_ylabel("MAPE (%)", fontsize=10)
    ax2.set_title("MAPE per Fold — Model A vs Model B", fontsize=11, color=BLUE)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.25, linestyle="--")
    ax2.spines[["top", "right"]].set_visible(False)

    # ── Panel 3: Aggregate bar chart ─────────────────────────────────────────
    ax3 = axes[2]
    metrics  = ["MAE", "RMSE", "MAPE (%)"]
    a_vals   = [fold_df["A_MAE"].mean(), fold_df["A_RMSE"].mean(), fold_df["A_MAPE"].mean()]
    b_vals   = [fold_df["B_MAE"].mean(), fold_df["B_RMSE"].mean(), fold_df["B_MAPE"].mean()]
    a_stds   = [fold_df["A_MAE"].std(),  fold_df["A_RMSE"].std(),  fold_df["A_MAPE"].std()]
    b_stds   = [fold_df["B_MAE"].std(),  fold_df["B_RMSE"].std(),  fold_df["B_MAPE"].std()]

    x = np.arange(len(metrics))
    w = 0.32
    bars_a = ax3.bar(x - w/2, a_vals, w, yerr=a_stds, capsize=4,
                     color=GREY,   label="Model A (lags)", edgecolor="white", alpha=0.85)
    bars_b = ax3.bar(x + w/2, b_vals, w, yerr=b_stds, capsize=4,
                     color=ORANGE, label="Model B (+ FluNet)", edgecolor="white")
    ax3.bar_label(bars_a, fmt="%.2f", padding=3, fontsize=8, color="#555")
    ax3.bar_label(bars_b, fmt="%.2f", padding=3, fontsize=8, color=ORANGE)
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics, fontsize=10)
    ax3.set_ylabel("Mean value (error bars = std)", fontsize=10)
    ax3.set_title(f"Aggregate Metrics — {len(fold_df)} Folds", fontsize=11, color=BLUE)
    ax3.legend(fontsize=9)
    ax3.grid(axis="y", alpha=0.25, linestyle="--")
    ax3.spines[["top", "right"]].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved plot: {out_path}")


def main():
    print("=" * 70)
    print("STEP 10: WALK-FORWARD VALIDATION")
    print("=" * 70)

    df = pd.read_csv(
        os.path.join(PROC_DIR, "integrated_dataset.csv"),
        parse_dates=["week_date"], index_col="week_date"
    )
    print(f"[OK] Loaded {len(df)} weeks ({df.index.min().date()} to {df.index.max().date()})")

    # Feature sets matching Step 7
    features_a = [f for f in ["R03_lag1", "R03_lag4_avg"] if f in df.columns]
    flu_cols   = [c for c in df.columns if "flu_" in c]
    features_b = features_a + flu_cols

    if not features_a:
        print("[ERROR] Lag features not found. Run steps 05-07 first.")
        return

    print(f"[OK] Model A features ({len(features_a)}): {features_a}")
    print(f"[OK] Model B features ({len(features_b)}): {features_b}")

    n_folds_expected = (len(df) - MIN_TRAIN) // STEP
    print(f"[OK] Expected folds: ~{n_folds_expected} "
          f"(min_train={MIN_TRAIN}w, step={STEP}w, test_window={TEST_WINDOW}w)")

    print("\n[...] Running walk-forward validation...")
    fold_df, pred_df = run_walk_forward(df, features_a, features_b)

    print(f"[OK] Completed {len(fold_df)} folds — "
          f"{len(pred_df)} total out-of-sample predictions")

    # ── Save outputs ──────────────────────────────────────────────────────────
    fold_df.to_csv(os.path.join(OUT_DIR, "wfv_fold_results.csv"), index=False)
    print("[OK] Saved: output/wfv_fold_results.csv")

    pred_df.to_csv(os.path.join(OUT_DIR, "wfv_predictions.csv"))
    print("[OK] Saved: output/wfv_predictions.csv")

    # Aggregate summary
    summary_rows = []
    for model, prefix in [("Model_A_lags_only", "A"), ("Model_B_lags_FluNet", "B")]:
        summary_rows.append({
            "model":      model,
            "MAPE_mean":  fold_df[f"{prefix}_MAPE"].mean(),
            "MAPE_std":   fold_df[f"{prefix}_MAPE"].std(),
            "MAPE_min":   fold_df[f"{prefix}_MAPE"].min(),
            "MAPE_max":   fold_df[f"{prefix}_MAPE"].max(),
            "MAE_mean":   fold_df[f"{prefix}_MAE"].mean(),
            "MAE_std":    fold_df[f"{prefix}_MAE"].std(),
            "RMSE_mean":  fold_df[f"{prefix}_RMSE"].mean(),
            "RMSE_std":   fold_df[f"{prefix}_RMSE"].std(),
            "R2_mean":    fold_df[f"{prefix}_R2"].mean(),
            "R2_std":     fold_df[f"{prefix}_R2"].std(),
            "n_folds":    len(fold_df),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUT_DIR, "wfv_summary.csv"), index=False)
    print("[OK] Saved: output/wfv_summary.csv")

    # Save as JSON for dashboard
    wfv_meta = {
        "min_train_weeks":  MIN_TRAIN,
        "step_weeks":       STEP,
        "test_window_weeks": TEST_WINDOW,
        "n_folds":          int(len(fold_df)),
        "n_predictions":    int(len(pred_df)),
        "A_MAPE_mean":      round(float(fold_df["A_MAPE"].mean()), 3),
        "A_MAPE_std":       round(float(fold_df["A_MAPE"].std()),  3),
        "B_MAPE_mean":      round(float(fold_df["B_MAPE"].mean()), 3),
        "B_MAPE_std":       round(float(fold_df["B_MAPE"].std()),  3),
        "B_wins_pct":       round(float((fold_df["B_MAPE"] < fold_df["A_MAPE"]).mean() * 100), 1),
        "delta_mape_mean":  round(float((fold_df["A_MAPE"] - fold_df["B_MAPE"]).mean()), 3),
    }
    with open(os.path.join(OUT_DIR, "wfv_meta.json"), "w") as f:
        json.dump(wfv_meta, f, indent=2)
    print("[OK] Saved: output/wfv_meta.json")

    print_summary(fold_df)

    # Highlight most volatile folds (high MAPE)
    top_hard = fold_df.nlargest(3, "B_MAPE")[
        ["fold", "test_start", "test_end", "B_MAPE", "A_MAPE", "train_weeks"]
    ]
    print("\n  Hardest folds for Model B (highest MAPE):")
    for _, row in top_hard.iterrows():
        print(f"    Fold {int(row.fold):3d}  ({row.test_start} -> {row.test_end})  "
              f"B_MAPE={row.B_MAPE:.1f}%  A_MAPE={row.A_MAPE:.1f}%  "
              f"train_w={int(row.train_weeks)}")

    top_easy = fold_df.nsmallest(3, "B_MAPE")[
        ["fold", "test_start", "test_end", "B_MAPE", "A_MAPE"]
    ]
    print("\n  Easiest folds for Model B (lowest MAPE):")
    for _, row in top_easy.iterrows():
        print(f"    Fold {int(row.fold):3d}  ({row.test_start} -> {row.test_end})  "
              f"B_MAPE={row.B_MAPE:.1f}%  A_MAPE={row.A_MAPE:.1f}%")

    plot_wfv(fold_df, pred_df, os.path.join(OUT_DIR, "wfv_plot.png"))

    print("\n[DONE] Walk-forward validation complete.")
    print("       Outputs: wfv_fold_results.csv | wfv_predictions.csv | "
          "wfv_summary.csv | wfv_meta.json | wfv_plot.png")


if __name__ == "__main__":
    main()
