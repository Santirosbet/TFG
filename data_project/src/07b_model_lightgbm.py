"""
Step 07b — LightGBM Model
Alternativa a XGBoost. Misma arquitectura de evaluación para comparación justa.

Modelos:
  LGB-A : features autorregresivos solo         (=XGBoost Model A)
  LGB-B : + señal flu australiana               (=XGBoost Model B)
  LGB-C : + señales nuevas de dataset_v2        (Google Trends + temperatura)

Evaluación: misma WFV de 48 folds, split cronológico estricto, sin shuffling.
Comparación final: DM test LightGBM-B vs XGBoost-B.

Outputs:
  output/lgbm_fold_results.csv    — métricas por fold (A, B, C)
  output/lgbm_predictions.csv     — predicciones OOS de los 3 modelos
  output/lgbm_meta.json           — resumen de métricas + DM test
  output/lgbm_comparison_plot.png — comparativa LightGBM vs XGBoost

Run: python src/07b_model_lightgbm.py
"""

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    MAIN_DATASET, ENHANCED_DATASET, OUTPUT_DIR, RANDOM_SEED,
    WFV_FOLDS, LGBM_META
)

warnings.filterwarnings("ignore")

# ── WFV parámetros — idénticos al Step 10 ──────────────────────────────────
MIN_TRAIN   = 104
STEP        = 4
TEST_WINDOW = 4

BLUE   = "#1f4e79"
ORANGE = "#ff6b35"
GREEN  = "#70ad47"
PURPLE = "#7030a0"
GREY   = "#aaaaaa"


def build_lgbm(extra_params: dict | None = None):
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("LightGBM no instalado. Ejecuta: pip install lightgbm")

    params = {
        "n_estimators":   300,
        "max_depth":      4,
        "learning_rate":  0.05,
        "subsample":      0.8,
        "colsample_bytree": 0.8,
        "random_state":   RANDOM_SEED,
        "verbosity":      -1,
        "n_jobs":         -1,
    }
    if extra_params:
        params.update(extra_params)
    return lgb.LGBMRegressor(**params)


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def dm_test(e1: np.ndarray, e2: np.ndarray) -> tuple[float, float]:
    """
    Diebold-Mariano test. e1 = errores modelo 1, e2 = errores modelo 2.
    H0: igual precisión. stat < 0 → modelo 2 es mejor.
    """
    d = e1 - e2
    n = len(d)
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)
    if var_d == 0:
        return 0.0, 1.0
    stat = mean_d / np.sqrt(var_d / n)
    pval = 2 * (1 - stats.norm.cdf(abs(stat)))
    return float(stat), float(pval)


def run_wfv(df: pd.DataFrame, feature_sets: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Walk-forward validation expandible — misma lógica que Step 10.

    feature_sets: {'A': [...], 'B': [...], 'C': [...]}
    """
    n = len(df)
    target = "R03"
    fold_records = []
    all_preds = []
    fold_idx = 0
    train_end = MIN_TRAIN

    model_keys = list(feature_sets.keys())

    while train_end + TEST_WINDOW <= n:
        test_start = train_end
        test_end   = train_end + TEST_WINDOW

        y_tr   = df.iloc[:train_end][target]
        y_te   = df.iloc[test_start:test_end][target]
        dates  = df.index[test_start:test_end]
        y_arr  = y_te.values

        fold_preds = {"actual_R03": y_arr, "week_date": dates, "fold": fold_idx + 1}
        fold_rec = {
            "fold":       fold_idx + 1,
            "train_weeks": train_end,
            "test_start": dates[0].strftime("%Y-%m-%d"),
            "test_end":   dates[-1].strftime("%Y-%m-%d"),
        }

        for key, features in feature_sets.items():
            feats_available = [f for f in features if f in df.columns]
            if not feats_available:
                continue
            X_tr = df.iloc[:train_end][feats_available]
            X_te = df.iloc[test_start:test_end][feats_available]

            model = build_lgbm()
            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)

            fold_preds[f"pred_lgbm_{key}"] = preds
            fold_rec[f"LGB{key}_MAE"]  = float(mean_absolute_error(y_arr, preds))
            fold_rec[f"LGB{key}_RMSE"] = float(np.sqrt(mean_squared_error(y_arr, preds)))
            fold_rec[f"LGB{key}_MAPE"] = compute_mape(y_arr, preds)
            fold_rec[f"LGB{key}_R2"]   = float(r2_score(y_arr, preds))

        fold_records.append(fold_rec)

        for i, dt in enumerate(dates):
            row = {"week_date": dt, "fold": fold_idx + 1, "train_weeks": train_end,
                   "actual_R03": float(y_arr[i])}
            for key in model_keys:
                if f"pred_lgbm_{key}" in fold_preds:
                    row[f"pred_lgbm_{key}"] = float(fold_preds[f"pred_lgbm_{key}"][i])
            all_preds.append(row)

        fold_idx  += 1
        train_end += STEP

    fold_df = pd.DataFrame(fold_records)
    pred_df = pd.DataFrame(all_preds).set_index("week_date")
    return fold_df, pred_df


def load_xgb_predictions() -> pd.DataFrame | None:
    """Carga las predicciones XGBoost del Step 10 para comparación DM."""
    xgb_path = OUTPUT_DIR / "wfv_predictions.csv"
    if not xgb_path.exists():
        print(f"[WARN] {xgb_path} no encontrado — DM test vs XGBoost no disponible")
        return None
    df = pd.read_csv(xgb_path, parse_dates=["week_date"], index_col="week_date")
    return df


def plot_comparison(fold_df: pd.DataFrame, pred_df: pd.DataFrame, xgb_preds: pd.DataFrame | None):
    n_models = sum(1 for c in fold_df.columns if c.endswith("_MAPE"))
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle("LightGBM vs XGBoost — Comparativa Walk-Forward Validation\n"
                 f"Expanding window | {len(fold_df)} folds | min_train={MIN_TRAIN}w",
                 fontsize=12, fontweight="bold", color=BLUE)

    # Panel 1: predicciones OOS
    ax = axes[0]
    ax.plot(pred_df.index, pred_df["actual_R03"],
            color="black", lw=2, label="Actual R03", zorder=5)
    if "pred_lgbm_B" in pred_df.columns:
        ax.plot(pred_df.index, pred_df["pred_lgbm_B"],
                color=ORANGE, lw=1.5, linestyle="--", alpha=0.85,
                label=f"LGB-B (MAPE={fold_df['LGBB_MAPE'].mean():.1f}%)")
    if xgb_preds is not None and "pred_B" in xgb_preds.columns:
        aligned = xgb_preds["pred_B"].reindex(pred_df.index)
        xgb_mape_str = ""
        xgb_meta = OUTPUT_DIR / "wfv_meta.json"
        if xgb_meta.exists():
            with open(xgb_meta) as f:
                m = json.load(f)
            xgb_mape_str = f"MAPE={m.get('B_MAPE_mean', '?'):.1f}%"
        ax.plot(pred_df.index, aligned, color=BLUE, lw=1.5, linestyle=":",
                alpha=0.7, label=f"XGB-B ({xgb_mape_str})")
    ax.set_ylabel("R03 units / week", fontsize=9)
    ax.set_title("Predicciones OOS: LightGBM-B vs XGBoost-B", fontsize=10, color=BLUE)
    ax.legend(fontsize=8); ax.grid(alpha=0.2, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    # Panel 2: MAPE por fold
    ax2 = axes[1]
    folds = fold_df["fold"]
    colors_keys = [("A", GREY), ("B", ORANGE), ("C", PURPLE)]
    for key, color in colors_keys:
        col = f"LGB{key}_MAPE"
        if col in fold_df.columns:
            mean_mape = fold_df[col].mean()
            ax2.plot(folds, fold_df[col], color=color, lw=1.3, marker="o", ms=3,
                     label=f"LGB-{key} (mean={mean_mape:.1f}%)", alpha=0.85)
            ax2.axhline(mean_mape, color=color, linestyle=":", lw=1, alpha=0.6)

    if xgb_preds is not None:
        xgb_meta_path = OUTPUT_DIR / "wfv_meta.json"
        if xgb_meta_path.exists():
            with open(xgb_meta_path) as f:
                m = json.load(f)
            ax2.axhline(m.get("B_MAPE_mean", 44.16), color=BLUE, linestyle="--",
                        lw=2, label=f"XGB-B mean (ref: {m.get('B_MAPE_mean', 44.16):.1f}%)")

    ax2.set_xlabel("Fold", fontsize=9)
    ax2.set_ylabel("MAPE (%)", fontsize=9)
    ax2.set_title("MAPE por fold — LightGBM A / B / C", fontsize=10, color=BLUE)
    ax2.legend(fontsize=8); ax2.grid(alpha=0.2, linestyle="--")
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = OUTPUT_DIR / "lgbm_comparison_plot.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Plot guardado: {out}")


def main():
    print("=" * 65)
    print("STEP 07b: LIGHTGBM MODEL — WFV + DM TEST vs XGBOOST")
    print("=" * 65)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Cargar datasets ---
    if not MAIN_DATASET.exists():
        print(f"[ERROR] Dataset principal no encontrado: {MAIN_DATASET}")
        print("  Ejecuta run_pipeline.py (steps 1-5) primero.")
        return

    df_base = pd.read_csv(MAIN_DATASET, parse_dates=["week_date"], index_col="week_date").sort_index()
    print(f"[OK] Dataset base: {df_base.shape}")

    # Dataset v2 (puede no existir aún)
    df_v2 = None
    if ENHANCED_DATASET.exists():
        df_v2 = pd.read_csv(ENHANCED_DATASET, parse_dates=["week_date"], index_col="week_date").sort_index()
        print(f"[OK] Dataset v2: {df_v2.shape}")
    else:
        print(f"[WARN] Dataset v2 no encontrado. LGB-C se omitirá.")
        print(f"       Ejecuta src/05b_integrate_v2.py para generarlo.")

    # --- Definir feature sets ---
    features_a = [f for f in ["R03_lag1", "R03_lag4_avg"] if f in df_base.columns]
    flu_cols   = [c for c in df_base.columns if "flu_" in c]
    features_b = features_a + flu_cols

    feature_sets = {"A": features_a, "B": features_b}

    if df_v2 is not None:
        new_signal_cols = [c for c in df_v2.columns if any(
            kw in c for kw in ["trends_", "temp_europe", "humidity_europe"]
        )]
        if new_signal_cols:
            # Añadir nuevas features al df_base para el modelo C
            for col in new_signal_cols:
                df_base[col] = df_v2[col].reindex(df_base.index)
            features_c = features_b + new_signal_cols
            feature_sets["C"] = features_c
            print(f"[OK] LGB-C features nuevas: {new_signal_cols}")
        else:
            print("[WARN] No se encontraron nuevas features en dataset_v2.")

    print(f"\n[OK] Modelos a entrenar: {list(feature_sets.keys())}")
    print(f"  LGB-A: {len(features_a)} features")
    print(f"  LGB-B: {len(features_b)} features")
    if "C" in feature_sets:
        print(f"  LGB-C: {len(feature_sets['C'])} features")

    n_expected = (len(df_base) - MIN_TRAIN) // STEP
    print(f"\n[...] Ejecutando WFV (~{n_expected} folds)...")

    fold_df, pred_df = run_wfv(df_base, feature_sets)
    print(f"[OK] WFV completado: {len(fold_df)} folds, {len(pred_df)} predicciones OOS")

    # --- Guardar outputs ---
    fold_path = OUTPUT_DIR / "lgbm_fold_results.csv"
    fold_df.to_csv(fold_path, index=False)
    print(f"[OK] {fold_path}")

    pred_path = OUTPUT_DIR / "lgbm_predictions.csv"
    pred_df.to_csv(pred_path)
    print(f"[OK] {pred_path}")

    # --- DM test vs XGBoost ---
    xgb_preds = load_xgb_predictions()
    dm_results = {}

    if xgb_preds is not None and "pred_B" in xgb_preds.columns:
        # Alinear predicciones
        aligned = pred_df.join(xgb_preds[["actual_R03", "pred_B"]], rsuffix="_xgb", how="inner")
        if len(aligned) > 10 and "pred_lgbm_B" in aligned.columns:
            e_xgb = np.abs(aligned["actual_R03"] - aligned["pred_B"])
            e_lgb = np.abs(aligned["actual_R03"] - aligned["pred_lgbm_B"])
            stat, pval = dm_test(e_xgb, e_lgb)
            dm_results["lgbm_B_vs_xgb_B"] = {
                "dm_stat": round(stat, 4),
                "p_value": round(pval, 4),
                "significant": pval < 0.05,
                "better_model": "LightGBM-B" if stat > 0 else "XGBoost-B",
                "note": "stat>0 → LGB mejor; stat<0 → XGB mejor"
            }
            print(f"\n  DM test LGB-B vs XGB-B: stat={stat:.3f}, p={pval:.3f} "
                  f"({'*' if pval < 0.05 else 'n.s.'})")
            print(f"  Mejor modelo: {dm_results['lgbm_B_vs_xgb_B']['better_model']}")

    # --- Meta JSON ---
    meta = {
        "n_folds":      int(len(fold_df)),
        "min_train":    MIN_TRAIN,
        "step":         STEP,
        "test_window":  TEST_WINDOW,
        "models":       {},
        "dm_tests":     dm_results,
        "xgboost_reference_mape": 44.16,
    }

    print("\n--- RESUMEN DE MÉTRICAS ---")
    for key in ["A", "B", "C"]:
        col = f"LGB{key}_MAPE"
        if col in fold_df.columns:
            mean_mape = float(fold_df[col].mean())
            std_mape  = float(fold_df[col].std())
            mean_mae  = float(fold_df[f"LGB{key}_MAE"].mean())
            mean_r2   = float(fold_df[f"LGB{key}_R2"].mean())
            meta["models"][f"LGB-{key}"] = {
                "MAPE_mean": round(mean_mape, 3),
                "MAPE_std":  round(std_mape, 3),
                "MAE_mean":  round(mean_mae, 3),
                "R2_mean":   round(mean_r2, 4),
            }
            print(f"  LGB-{key}: MAPE={mean_mape:.2f}% (±{std_mape:.2f}%)  "
                  f"MAE={mean_mae:.2f}  R²={mean_r2:.4f}")

    xgb_ref = 44.16
    if "LGB-B" in meta["models"]:
        lgb_b_mape = meta["models"]["LGB-B"]["MAPE_mean"]
        improvement = xgb_ref - lgb_b_mape
        meta["lgbm_B_vs_xgb_B_mape_delta"] = round(improvement, 3)
        sign = "mejor" if improvement > 0 else "peor"
        print(f"\n  LGB-B vs XGB-B: {improvement:+.2f}pp ({sign} que XGBoost)")
        print(f"  XGBoost-B referencia: {xgb_ref}%")

    with open(LGBM_META, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[OK] Meta guardado: {LGBM_META}")

    # --- Plot ---
    plot_comparison(fold_df, pred_df, xgb_preds)

    print("\n[DONE] Step 07b completo.")


if __name__ == "__main__":
    main()
