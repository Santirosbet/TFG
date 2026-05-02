"""
Step 7: Train XGBoost model on the integrated dataset.
Compares baseline (synthetic proxy) vs improved (real data) approaches.
Output: output/model_results.png, console metrics

Run: python src/07_model_xgboost.py
"""

import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = compute_mape(y_true, y_pred)

    print(f"\n--- {model_name} ---")
    print(f"  MAE:  {mae:.2f} units")
    print(f"  RMSE: {rmse:.2f} units")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R2:   {r2:.4f}")

    return {"model": model_name, "MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


def train_and_evaluate(X_train, y_train, X_test, y_test, model_name):
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = evaluate_model(y_test, predictions, model_name)
    return model, predictions, metrics


def main():
    print("=" * 60)
    print("STEP 7: XGBOOST MODEL TRAINING & COMPARISON")
    print("=" * 60)

    # Load integrated dataset
    path = os.path.join(PROC_DIR, "integrated_dataset.csv")
    if not os.path.exists(path):
        print("[ERROR] Integrated dataset not found. Run step 05 first.")
        return

    df = pd.read_csv(path, parse_dates=["week_date"], index_col="week_date")
    print(f"[OK] Loaded {len(df)} rows, columns: {list(df.columns)}")

    # Chronological train/test split (80/20)
    split_idx = int(len(df) * 0.8)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    print(f"[INFO] Train: {len(train)} weeks ({train.index.min().date()} to {train.index.max().date()})")
    print(f"[INFO] Test:  {len(test)} weeks ({test.index.min().date()} to {test.index.max().date()})")

    target = "R03"
    y_train = train[target]
    y_test = test[target]

    all_metrics = []

    # --- Model A: Baseline (lag features only, no real Australian data) ---
    baseline_features = ["R03_lag1", "R03_lag4_avg"]
    available_baseline = [f for f in baseline_features if f in df.columns]
    if available_baseline:
        model_a, pred_a, metrics_a = train_and_evaluate(
            train[available_baseline], y_train,
            test[available_baseline], y_test,
            "Baseline (Lag features only)"
        )
        all_metrics.append(metrics_a)

    # --- Model B: With real FluNet data ---
    flu_features = [c for c in df.columns if "flu_" in c]
    if flu_features:
        model_b_features = available_baseline + flu_features
        model_b, pred_b, metrics_b = train_and_evaluate(
            train[model_b_features], y_train,
            test[model_b_features], y_test,
            "With FluNet Epidemiological Data"
        )
        all_metrics.append(metrics_b)

    # --- Model C: Full model (all features including real Australian pharma) ---
    au_features = [c for c in df.columns if "au_" in c and c not in flu_features]
    if au_features:
        model_c_features = available_baseline + flu_features + au_features
        model_c, pred_c, metrics_c = train_and_evaluate(
            train[model_c_features], y_train,
            test[model_c_features], y_test,
            "Full Model (Lags + FluNet + Real AU Pharma)"
        )
        all_metrics.append(metrics_c)

    # --- Comparison table ---
    if all_metrics:
        print("\n" + "=" * 60)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 60)
        results_df = pd.DataFrame(all_metrics)
        print(results_df.to_string(index=False))

        # Save results
        results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison_v3.csv"), index=False)

    # --- Save best model (FluNet model, or baseline if FluNet not available) ---
    best_model = model_b if flu_features else model_a
    best_features = model_b_features if flu_features else available_baseline
    model_path = os.path.join(OUTPUT_DIR, "xgboost_best_model.json")
    best_model.save_model(model_path)
    print(f"\n[OK] Saved best model: {model_path}")

    # Save feature list and split metadata for dashboard
    import json
    meta = {
        "features": best_features,
        "target": target,
        "train_end": str(train.index.max().date()),
        "test_start": str(test.index.min().date()),
        "test_end": str(test.index.max().date()),
        "split_idx": split_idx,
    }
    with open(os.path.join(OUTPUT_DIR, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[OK] Saved model metadata: output/model_meta.json")

    # --- Visualization: best model predictions vs actual ---
    best_pred = pred_b if flu_features else pred_a
    best_name = "With FluNet" if flu_features else "Baseline"

    # Save test predictions for dashboard (needs best_pred defined above)
    pred_df = pd.DataFrame({
        "week_date": test.index,
        "actual_R03": y_test.values,
        "predicted_R03": best_pred,
    })
    pred_df.to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False)
    print(f"[OK] Saved test predictions: output/test_predictions.csv")

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Actual vs Predicted
    ax1 = axes[0]
    ax1.plot(test.index, y_test.values, label="Actual R03 Demand", color="black", linewidth=2)
    ax1.plot(test.index, best_pred, label=f"Predicted ({best_name})", color="red", linestyle="--", alpha=0.8)
    ax1.set_title("R03 Demand: Actual vs Predicted (Test Set)", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Units Sold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Comparison bar chart
    if len(all_metrics) > 1:
        ax2 = axes[1]
        models = [m["model"][:25] for m in all_metrics]
        x = np.arange(len(models))
        width = 0.25

        mae_vals = [m["MAE"] for m in all_metrics]
        rmse_vals = [m["RMSE"] for m in all_metrics]
        mape_vals = [m["MAPE"] for m in all_metrics]

        ax2.bar(x - width, mae_vals, width, label="MAE", color="#1f77b4")
        ax2.bar(x, rmse_vals, width, label="RMSE", color="#ff7f0e")
        ax2.bar(x + width, mape_vals, width, label="MAPE (%)", color="#2ca02c")

        ax2.set_title("Model Comparison: Error Metrics", fontsize=14, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, fontsize=9)
        ax2.legend()
        ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "model_results.png")
    plt.savefig(fig_path, dpi=150)
    print(f"\n[OK] Saved visualization: {fig_path}")
    plt.close()

    print("\n[DONE] Model training and comparison complete.")


if __name__ == "__main__":
    main()
