"""
Step 16b — Stacking Ensemble con Meta-Learner (Ridge)
Reemplaza el ensemble de pesos iguales (que empeoró resultados) con un
meta-learner que aprende cuándo confiar en cada modelo.

Arquitectura:
  Base models : XGBoost Model B + LightGBM Model B
  Meta-learner: Ridge Regression (simple, interpretable, evita overfitting)
  Meta-features: predicciones OOF (out-of-fold) de los modelos base

Metodología:
  1. WFV con ventana expandible — idéntica al Step 10.
  2. En cada fold, entrenamos los dos base models en el train set.
  3. Guardamos las predicciones OOF del test set.
  4. Con TODAS las OOF acumuladas, entrenamos el meta-learner (split temporal).
  5. El meta-learner predice la demanda final a partir de las predicciones base.
  6. Separación temporal estricta: el meta-learner NUNCA ve datos futuros.

NOTA: Si el stacking no mejora, se documenta honestamente (como el ensemble AU+NZ).

Outputs:
  output/stacking_predictions.csv  — predicciones OOS del ensemble
  output/stacking_meta.json        — métricas + DM test vs XGBoost-B solo
  output/stacking_plot.png         — visualización

Run: python src/16b_stacking_ensemble.py
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
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    MAIN_DATASET, OUTPUT_DIR, RANDOM_SEED, STACKING_META
)

warnings.filterwarnings("ignore")

MIN_TRAIN   = 104   # igual que Step 10
STEP        = 4
TEST_WINDOW = 4
META_TRAIN_FOLDS = 20  # primeros N folds para entrenar el meta-learner

BLUE   = "#1f4e79"
ORANGE = "#ff6b35"
GREEN  = "#70ad47"
PURPLE = "#7030a0"
GREY   = "#aaaaaa"


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def dm_test(e1: np.ndarray, e2: np.ndarray) -> tuple[float, float]:
    d = e1 - e2
    n = len(d)
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)
    if var_d == 0:
        return 0.0, 1.0
    stat = mean_d / np.sqrt(var_d / n)
    pval = 2 * (1 - stats.norm.cdf(abs(stat)))
    return float(stat), float(pval)


def build_xgb():
    import xgboost as xgb
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
    )


def build_lgbm():
    try:
        import lightgbm as lgb
        return lgb.LGBMRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_SEED,
            verbosity=-1,
        )
    except ImportError:
        print("[WARN] LightGBM no instalado — stacking solo usará XGBoost")
        return None


def run_stacking_wfv(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    WFV con stacking en dos fases:
      Fase 1: Generar OOF predictions de XGB-B y LGB-B.
      Fase 2: Entrenar meta-learner Ridge en las primeras META_TRAIN_FOLDS OOF,
              y predecir en el resto.

    Devuelve DataFrame con predicciones OOS de todos los modelos + stacking.
    """
    n = len(df)
    target = "R03"
    oof_records = []  # Acumular OOF para entrenar meta-learner

    fold_idx  = 0
    train_end = MIN_TRAIN

    lgbm_model = build_lgbm()
    use_lgbm = lgbm_model is not None

    print(f"  Base models: XGBoost{'  + LightGBM' if use_lgbm else ''}")
    print(f"  Meta-learner: Ridge Regression")
    print(f"  Meta-training folds: {META_TRAIN_FOLDS} (luego predicción)")

    while train_end + TEST_WINDOW <= n:
        test_start = train_end
        test_end   = train_end + TEST_WINDOW

        X_tr = df.iloc[:train_end][features]
        y_tr = df.iloc[:train_end][target]
        X_te = df.iloc[test_start:test_end][features]
        y_te = df.iloc[test_start:test_end][target]
        dates = df.index[test_start:test_end]

        # XGBoost prediction
        xgb_m = build_xgb()
        xgb_m.fit(X_tr, y_tr)
        p_xgb = xgb_m.predict(X_te)

        # LightGBM prediction
        p_lgbm = None
        if use_lgbm:
            lgbm_model_fold = build_lgbm()
            lgbm_model_fold.fit(X_tr, y_tr)
            p_lgbm = lgbm_model_fold.predict(X_te)

        for i, dt in enumerate(dates):
            rec = {
                "week_date":   dt,
                "fold":        fold_idx + 1,
                "train_weeks": train_end,
                "actual_R03":  float(y_te.values[i]),
                "pred_xgb":    float(p_xgb[i]),
                "pred_lgbm":   float(p_lgbm[i]) if p_lgbm is not None else float(p_xgb[i]),
            }
            oof_records.append(rec)

        fold_idx  += 1
        train_end += STEP

    oof_df = pd.DataFrame(oof_records).set_index("week_date")
    oof_df = oof_df.sort_index()

    print(f"\n  OOF generadas: {len(oof_df)} predicciones en {fold_idx} folds")

    # ── Fase 2: Meta-learner (temporal split) ────────────────────────────────
    # El meta-learner se entrena en los primeros META_TRAIN_FOLDS folds
    # y predice en el resto. Separación temporal estricta.
    meta_train_mask = oof_df["fold"] <= META_TRAIN_FOLDS
    meta_test_mask  = oof_df["fold"] > META_TRAIN_FOLDS

    meta_features = ["pred_xgb", "pred_lgbm"]

    if meta_train_mask.sum() < 20:
        print("[WARN] Muy pocas muestras para entrenar meta-learner. Usando promedio simple.")
        oof_df["pred_stacked"] = (oof_df["pred_xgb"] + oof_df["pred_lgbm"]) / 2
    else:
        X_meta_train = oof_df.loc[meta_train_mask, meta_features].values
        y_meta_train = oof_df.loc[meta_train_mask, "actual_R03"].values
        X_meta_test  = oof_df.loc[meta_test_mask,  meta_features].values

        scaler = StandardScaler()
        X_meta_train_sc = scaler.fit_transform(X_meta_train)
        X_meta_test_sc  = scaler.transform(X_meta_test)

        ridge = Ridge(alpha=1.0, random_state=RANDOM_SEED)
        ridge.fit(X_meta_train_sc, y_meta_train)

        print(f"\n  Ridge meta-learner coeficientes:")
        print(f"    XGB coef:  {ridge.coef_[0]:.4f}")
        if use_lgbm:
            print(f"    LGBM coef: {ridge.coef_[1]:.4f}")
        print(f"    Intercept: {ridge.intercept_:.4f}")

        oof_df["pred_stacked"] = np.nan
        oof_df.loc[meta_test_mask, "pred_stacked"] = ridge.predict(X_meta_test_sc)
        # En meta-train folds, usar promedio simple (no hay predicción real del meta-learner)
        oof_df.loc[meta_train_mask, "pred_stacked"] = (
            oof_df.loc[meta_train_mask, "pred_xgb"] +
            oof_df.loc[meta_train_mask, "pred_lgbm"]
        ) / 2

    return oof_df


def plot_stacking(oof_df: pd.DataFrame, metrics: dict):
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle("Stacking Ensemble — Ridge Meta-Learner\n"
                 "Base: XGBoost-B + LightGBM-B | Meta: Ridge Regression (OOF)",
                 fontsize=12, fontweight="bold", color=BLUE)

    ax = axes[0]
    ax.plot(oof_df.index, oof_df["actual_R03"],
            color="black", lw=2, label="Actual R03", zorder=5)
    ax.plot(oof_df.index, oof_df["pred_xgb"],
            color=GREY, lw=1.2, linestyle="--", alpha=0.7,
            label=f"XGBoost-B (MAPE={metrics['xgb_mape']:.1f}%)")
    if "lgbm_mape" in metrics:
        ax.plot(oof_df.index, oof_df["pred_lgbm"],
                color=ORANGE, lw=1.2, linestyle="--", alpha=0.7,
                label=f"LightGBM-B (MAPE={metrics['lgbm_mape']:.1f}%)")
    if oof_df["pred_stacked"].notna().sum() > 10:
        ax.plot(oof_df.index, oof_df["pred_stacked"],
                color=PURPLE, lw=2,
                label=f"Stacking (MAPE={metrics.get('stacking_mape', '?'):.1f}%)")

    ax.set_ylabel("R03 units / week", fontsize=9)
    ax.set_title("Predicciones OOS: XGBoost vs LightGBM vs Stacking", fontsize=10, color=BLUE)
    ax.legend(fontsize=8); ax.grid(alpha=0.2, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    ax2 = axes[1]
    model_names = ["XGBoost-B", "LightGBM-B", "Stacking"]
    mape_vals   = [
        metrics.get("xgb_mape", np.nan),
        metrics.get("lgbm_mape", np.nan),
        metrics.get("stacking_mape", np.nan),
    ]
    colors = [GREY, ORANGE, PURPLE]
    bars = ax2.bar(model_names, mape_vals, color=colors, alpha=0.85, edgecolor="white")
    ax2.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=10)
    ax2.set_ylabel("MAPE (%)", fontsize=9)
    ax2.set_title("Comparativa MAPE — Baseline XGBoost vs Stacking", fontsize=10, color=BLUE)
    ax2.axhline(35.78, color=GREEN, linestyle=":", lw=2, label="Switching Rule (35.78%)")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.2, linestyle="--")
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = OUTPUT_DIR / "stacking_plot.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Plot guardado: {out}")


def main():
    print("=" * 65)
    print("STEP 16b: STACKING ENSEMBLE — RIDGE META-LEARNER")
    print("=" * 65)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MAIN_DATASET.exists():
        print(f"[ERROR] Dataset principal no encontrado: {MAIN_DATASET}")
        return

    df = pd.read_csv(MAIN_DATASET, parse_dates=["week_date"], index_col="week_date").sort_index()
    print(f"[OK] Dataset: {df.shape}")

    features_a = [f for f in ["R03_lag1", "R03_lag4_avg"] if f in df.columns]
    flu_cols   = [c for c in df.columns if "flu_" in c]
    features_b = features_a + flu_cols

    print(f"[OK] Features Model-B: {features_b}")
    print(f"\n[...] Ejecutando WFV con stacking...")

    oof_df = run_stacking_wfv(df, features_b)

    # --- Métricas ---
    actual = oof_df["actual_R03"].values
    metrics = {}

    metrics["xgb_mape"]  = compute_mape(actual, oof_df["pred_xgb"].values)
    metrics["lgbm_mape"] = compute_mape(actual, oof_df["pred_lgbm"].values)

    stacked_mask = oof_df["pred_stacked"].notna()
    if stacked_mask.sum() > 10:
        metrics["stacking_mape"] = compute_mape(
            actual[stacked_mask.values], oof_df["pred_stacked"].values[stacked_mask.values]
        )
    else:
        metrics["stacking_mape"] = np.nan

    print(f"\n--- RESULTADOS ---")
    print(f"  XGBoost-B  : MAPE={metrics['xgb_mape']:.2f}%")
    print(f"  LightGBM-B : MAPE={metrics['lgbm_mape']:.2f}%")
    print(f"  Stacking   : MAPE={metrics.get('stacking_mape', float('nan')):.2f}%")
    print(f"  Switching Rule (ref): 35.78%")

    # DM test: stacking vs XGBoost-B
    dm_results = {}
    if not np.isnan(metrics.get("stacking_mape", np.nan)):
        valid = stacked_mask.values
        e_xgb     = np.abs(actual[valid] - oof_df["pred_xgb"].values[valid])
        e_stacked = np.abs(actual[valid] - oof_df["pred_stacked"].values[valid])
        stat, pval = dm_test(e_xgb, e_stacked)
        dm_results["stacking_vs_xgb"] = {
            "dm_stat": round(stat, 4),
            "p_value": round(pval, 4),
            "significant": pval < 0.05,
            "better_model": "Stacking" if stat > 0 else "XGBoost-B",
        }
        print(f"\n  DM test Stacking vs XGB-B: stat={stat:.3f}, p={pval:.3f} "
              f"({'*' if pval < 0.05 else 'n.s.'})")
        print(f"  Mejor modelo: {dm_results['stacking_vs_xgb']['better_model']}")

    # Nota honesta si el stacking empeora
    if not np.isnan(metrics.get("stacking_mape", np.nan)):
        if metrics["stacking_mape"] > metrics["xgb_mape"]:
            metrics["honest_note"] = (
                "Stacking no mejora sobre XGBoost solo. "
                "Posible razón: los dos base models aprenden señales similares "
                "(alta correlación de predicciones OOF), reduciendo el beneficio del ensemble. "
                "Resultado negativo documentado honestamente."
            )
            print(f"\n  [RESULTADO NEGATIVO] Stacking no mejora sobre XGBoost solo.")
            print(f"  >> {metrics['honest_note']}")
        else:
            improvement = metrics["xgb_mape"] - metrics["stacking_mape"]
            metrics["improvement_pp"] = round(improvement, 3)
            print(f"\n  [RESULTADO POSITIVO] Stacking mejora {improvement:.2f}pp sobre XGBoost")

    # --- Guardar ---
    oof_df.to_csv(OUTPUT_DIR / "stacking_predictions.csv")
    print(f"\n[OK] Predicciones guardadas: output/stacking_predictions.csv")

    meta = {
        "n_observations": len(oof_df),
        "meta_train_folds": META_TRAIN_FOLDS,
        "meta_learner": "Ridge(alpha=1.0)",
        "base_models": ["XGBoost-B", "LightGBM-B"],
        "metrics": metrics,
        "dm_tests": dm_results,
        "switching_rule_reference_mape": 35.78,
    }
    with open(STACKING_META, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[OK] Meta guardado: {STACKING_META}")

    plot_stacking(oof_df, metrics)

    print("\n[DONE] Step 16b completo.")


if __name__ == "__main__":
    main()
