"""
Step 07c — XGBoost v3: Enhanced Features + Hyperparameter Tuning

Mejoras sobre XGBoost-B (Step 10):
  1. Codificacion ciclica de semana/mes (sin/cos) — el modelo entiende que
     semana 52 es casi igual a semana 1 (patron circular anual)
  2. Lags adicionales de R03: lag2, lag3, lag8, lag13 semanas
  3. Medias moviles: R03 rolling 4w, 8w, 12w — captura tendencia local
  4. Desviacion tipica rolling 4w — captura volatilidad reciente
  5. N02BE_lag1 (paracetamol semana anterior) — misma estacionalidad que R03,
     correlacion cruzada como predictor inter-categoria
  6. Señal australiana a 3 lags (24w, 26w, 28w) — cubre incertidumbre del lag exacto
  7. Hiperparametros optimizados para datasets pequeños:
     max_depth=4, learning_rate=0.05, n_estimators=500, subsample=0.8
  8. Switching rule aplicada al nuevo modelo (compara con original 35.78%)

Outputs:
  output/xgb_v3_meta.json         — metricas comparativas
  output/xgb_v3_predictions.csv   — predicciones WFV
  output/xgb_v3_plot.png          — comparacion visual

Run: python src/07c_model_xgboost_v3.py
"""

import sys
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    MAIN_DATASET, RAW_DIR, PROCESSED_DIR, OUTPUT_DIR,
    LAG_AU_FLU, SWITCHING_RULE_SUMMER_WEEKS, RANDOM_SEED
)

# ── WFV parameters (same as Step 10 for fair comparison) ──────────────────────
MIN_TRAIN   = 104
STEP        = 4
TEST_WINDOW = 4

BLUE   = "#1f4e79"
ORANGE = "#ff6b35"
GREEN  = "#70ad47"
RED    = "#c00000"
PURPLE = "#7030a0"


# ── Helpers ───────────────────────────────────────────────────────────────────
def mape(y_true, y_pred):
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def diebold_mariano(e1, e2):
    d = np.abs(e1) - np.abs(e2)
    n = len(d)
    if n < 2:
        return np.nan, np.nan
    dm_stat, p_value = stats.ttest_1samp(d, 0)
    return float(dm_stat), float(p_value)


def build_model_b():
    """Original XGBoost-B hyperparams (Step 10 baseline)."""
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=RANDOM_SEED,
        verbosity=0,
    )


def build_model_v3():
    """Enhanced hyperparams: shallower trees, lower LR, more estimators."""
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        verbosity=0,
    )


# ── Feature engineering ───────────────────────────────────────────────────────
def build_features(base_df, n02be_series, flunet_au_series):
    """
    Construye el dataset enriquecido con todas las nuevas features.

    Parametros
    ----------
    base_df        : integrated_dataset.csv (ya indexado por fecha)
    n02be_series   : serie semanal de ventas N02BE (paracetamol)
    flunet_au_series : serie semanal de INF_ALL australia (sin lag aplicado)
    """
    df = base_df.copy()

    # 1. Codificacion ciclica de semana del ano
    week = df.index.isocalendar().week.astype(float)
    df["week_sin"] = np.sin(2 * np.pi * week / 52.18)
    df["week_cos"] = np.cos(2 * np.pi * week / 52.18)

    # 2. Lags adicionales de R03
    df["R03_lag2"]  = df["R03"].shift(2)
    df["R03_lag3"]  = df["R03"].shift(3)
    df["R03_lag8"]  = df["R03"].shift(8)
    df["R03_lag13"] = df["R03"].shift(13)  # trimestral

    # 3. Rolling statistics de R03
    df["R03_rolling4_mean"]  = df["R03"].shift(1).rolling(4).mean()
    df["R03_rolling8_mean"]  = df["R03"].shift(1).rolling(8).mean()
    df["R03_rolling12_mean"] = df["R03"].shift(1).rolling(12).mean()
    df["R03_rolling4_std"]   = df["R03"].shift(1).rolling(4).std()

    # 4. N02BE lag1 (paracetamol semana anterior)
    n02be_aligned = n02be_series.reindex(df.index, method="nearest")
    df["N02BE_lag1"] = n02be_aligned.shift(1)

    # 5. Señal australiana a multiples lags
    flu_aligned = flunet_au_series.reindex(df.index, method="nearest")
    df["flu_au_lag24"] = flu_aligned.shift(24)
    df["flu_au_lag26"] = flu_aligned.shift(26)   # = flu_au_lagged ya existente
    df["flu_au_lag28"] = flu_aligned.shift(28)

    df = df.dropna()
    return df


# ── Walk-forward validation ───────────────────────────────────────────────────
def run_wfv(df, features_b, features_v3):
    """
    Expanding-window WFV.
    Compara Model B (original) vs Model V3 (enhanced features + hyperparams).
    """
    n = len(df)
    target = "R03"
    fold_records = []
    all_preds = []

    train_end = MIN_TRAIN
    fold_idx  = 0

    while train_end + TEST_WINDOW <= n:
        test_start = train_end
        test_end   = train_end + TEST_WINDOW

        X_tr_b  = df.iloc[:train_end][features_b]
        X_tr_v3 = df.iloc[:train_end][features_v3]
        y_tr    = df.iloc[:train_end][target]

        X_te_b  = df.iloc[test_start:test_end][features_b]
        X_te_v3 = df.iloc[test_start:test_end][features_v3]
        y_te    = df.iloc[test_start:test_end][target]
        dates   = df.index[test_start:test_end]

        m_b  = build_model_b();  m_b.fit(X_tr_b,  y_tr)
        m_v3 = build_model_v3(); m_v3.fit(X_tr_v3, y_tr)

        p_b  = m_b.predict(X_te_b)
        p_v3 = m_v3.predict(X_te_v3)
        y_arr = y_te.values

        fold_records.append({
            "fold":        fold_idx + 1,
            "train_weeks": train_end,
            "test_start":  dates[0].strftime("%Y-%m-%d"),
            "test_end":    dates[-1].strftime("%Y-%m-%d"),
            "B_MAPE":      mape(y_arr, p_b),
            "V3_MAPE":     mape(y_arr, p_v3),
            "B_MAE":       mean_absolute_error(y_arr, p_b),
            "V3_MAE":      mean_absolute_error(y_arr, p_v3),
        })

        for i, dt in enumerate(dates):
            all_preds.append({
                "date":      dt,
                "y_true":    y_arr[i],
                "y_pred_b":  float(p_b[i]),
                "y_pred_v3": float(p_v3[i]),
                "fold":      fold_idx + 1,
            })

        train_end += STEP
        fold_idx  += 1

    return pd.DataFrame(fold_records), pd.DataFrame(all_preds).set_index("date")


# ── Switching rule ─────────────────────────────────────────────────────────────
def apply_switching(preds_df, historical_seasonal_mean):
    """
    Para cada prediccion: usa el modelo en semanas de temporada alta,
    usa la media historica estacional en semanas de temporada baja.
    """
    preds = preds_df.copy()
    week = preds.index.isocalendar().week.astype(int)

    # V3 switching
    preds["y_switch_v3"] = preds["y_pred_v3"].copy()
    off_mask = week.isin(SWITCHING_RULE_SUMMER_WEEKS)
    for dt, row in preds[off_mask].iterrows():
        w = dt.isocalendar().week
        if w in historical_seasonal_mean.index:
            preds.loc[dt, "y_switch_v3"] = historical_seasonal_mean[w]

    # B switching (same as original step 17, for reference)
    preds["y_switch_b"] = preds["y_pred_b"].copy()
    for dt, row in preds[off_mask].iterrows():
        w = dt.isocalendar().week
        if w in historical_seasonal_mean.index:
            preds.loc[dt, "y_switch_b"] = historical_seasonal_mean[w]

    return preds


# ── Plot ───────────────────────────────────────────────────────────────────────
def make_plot(preds, mape_b, mape_v3, mape_sw_b, mape_sw_v3, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                             gridspec_kw={"height_ratios": [3, 1]})

    ax1 = axes[0]
    ax1.plot(preds.index, preds["y_true"],   color=BLUE,   lw=2,   label="Actual R03")
    ax1.plot(preds.index, preds["y_pred_b"], color=ORANGE, lw=1.5, linestyle="--",
             alpha=0.7, label=f"XGBoost-B  MAPE={mape_b:.1f}%")
    ax1.plot(preds.index, preds["y_pred_v3"],color=GREEN,  lw=1.5,
             label=f"XGBoost-V3 MAPE={mape_v3:.1f}%")
    if "y_switch_v3" in preds:
        ax1.plot(preds.index, preds["y_switch_v3"], color=PURPLE, lw=2,
                 label=f"V3 + Switch MAPE={mape_sw_v3:.1f}%")
    ax1.set_title("XGBoost v3 — Enhanced Features vs Baseline", fontsize=13, fontweight="bold")
    ax1.set_ylabel("R03 Units Sold")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    e_b  = np.abs(preds["y_true"] - preds["y_pred_b"])
    e_v3 = np.abs(preds["y_true"] - preds["y_pred_v3"])
    ax2.fill_between(preds.index, e_b,  alpha=0.5, color=ORANGE, label=f"Error B  (mean={e_b.mean():.1f})")
    ax2.fill_between(preds.index, e_v3, alpha=0.5, color=GREEN,  label=f"Error V3 (mean={e_v3.mean():.1f})")
    ax2.set_ylabel("Absolute Error")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Plot guardado: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("STEP 07c: XGBOOST V3 — ENHANCED FEATURES")
    print("=" * 65)

    # --- Cargar dataset base ---
    base = pd.read_csv(MAIN_DATASET, parse_dates=["week_date"], index_col="week_date")
    base = base.sort_index()
    print(f"[OK] Dataset base: {base.shape}")

    # --- Cargar N02BE de salesweekly ---
    sales_path = RAW_DIR / "salesweekly.csv"
    if not sales_path.exists():
        print(f"[WARN] {sales_path} no encontrado — N02BE_lag1 sera omitido")
        n02be = pd.Series(dtype=float, name="N02BE")
    else:
        sales = pd.read_csv(sales_path, parse_dates=["datum"], index_col="datum")
        sales.index.name = "week_date"
        n02be = sales["N02BE"].rename("N02BE")
        n02be = n02be.resample("W-SUN").mean()
        print(f"[OK] N02BE cargado: {len(n02be)} semanas")

    # --- Cargar señal bruta australiana (sin lag aplicado) ---
    flu_path = PROCESSED_DIR / "flunet_australia.csv"
    if not flu_path.exists():
        print(f"[WARN] {flu_path} no encontrado — usando flu_au_positives del dataset")
        flu_au_raw = base["flu_au_positives"]
    else:
        flu_raw = pd.read_csv(flu_path, parse_dates=["iso_date"]).set_index("iso_date")
        flu_raw.index.name = "week_date"
        flu_au_raw = flu_raw["INF_ALL"].resample("W-SUN").sum()
        print(f"[OK] FluNet Australia bruto cargado: {len(flu_au_raw)} semanas")

    # --- Feature engineering ---
    print("\n[1/4] Ingenieria de features...")
    df = build_features(base, n02be, flu_au_raw)
    print(f"[OK] Dataset enriquecido: {df.shape}")

    # Features Model B (original Step 10)
    features_b = ["R03_lag1", "R03_lag4_avg",
                  "flu_au_positives", "flu_au_lagged", "flu_eu_positives"]

    # Features V3 (todas las nuevas)
    features_v3 = [
        "R03_lag1", "R03_lag4_avg",                    # originales
        "flu_au_positives", "flu_au_lagged", "flu_eu_positives",
        "week_sin", "week_cos",                         # ciclico
        "R03_lag2", "R03_lag3", "R03_lag8", "R03_lag13",  # mas AR
        "R03_rolling4_mean", "R03_rolling8_mean",       # rolling
        "R03_rolling12_mean", "R03_rolling4_std",
        "N02BE_lag1",                                   # cross-drug
        "flu_au_lag24", "flu_au_lag26", "flu_au_lag28", # multi-lag flu
    ]

    # Filtrar a las que existen en el df
    features_v3 = [f for f in features_v3 if f in df.columns]
    print(f"  Features B:  {len(features_b)} features")
    print(f"  Features V3: {len(features_v3)} features")
    print(f"  Nuevas:      {[f for f in features_v3 if f not in features_b]}")

    # --- WFV ---
    print("\n[2/4] Walk-Forward Validation (48 folds)...")
    folds_df, preds_df = run_wfv(df, features_b, features_v3)
    print(f"[OK] WFV completado: {len(folds_df)} folds, {len(preds_df)} predicciones")

    mape_b  = folds_df["B_MAPE"].mean()
    mape_v3 = folds_df["V3_MAPE"].mean()

    print(f"\n  XGBoost-B  WFV MAPE: {mape_b:.3f}%")
    print(f"  XGBoost-V3 WFV MAPE: {mape_v3:.3f}%")
    print(f"  Delta: {mape_v3 - mape_b:+.3f}pp ({'mejor' if mape_v3 < mape_b else 'peor'})")

    # --- Switching Rule ---
    print("\n[3/4] Aplicando Switching Rule al V3...")
    # Calcular media historica estacional con todo el training previo al WFV
    train_hist = df.iloc[:MIN_TRAIN].copy()
    train_hist["week"] = train_hist.index.isocalendar().week.astype(int)
    hist_mean = train_hist.groupby("week")["R03"].mean()

    preds_sw = apply_switching(preds_df, hist_mean)
    preds_sw["week"] = preds_sw.index.isocalendar().week.astype(int)

    y_true = preds_sw["y_true"].values

    # MAPE switching V3
    sw_v3_preds = preds_sw["y_switch_v3"].values
    mape_sw_v3 = mape(y_true, sw_v3_preds)

    sw_b_preds = preds_sw["y_switch_b"].values
    mape_sw_b = mape(y_true, sw_b_preds)

    print(f"  XGBoost-B  + Switch MAPE: {mape_sw_b:.3f}%")
    print(f"  XGBoost-V3 + Switch MAPE: {mape_sw_v3:.3f}%")
    ORIGINAL_SWITCH_MAPE = 35.779
    print(f"  Switch original (Step 17): {ORIGINAL_SWITCH_MAPE:.3f}%")
    print(f"  Delta V3 vs original switch: {mape_sw_v3 - ORIGINAL_SWITCH_MAPE:+.3f}pp")

    # --- Diebold-Mariano ---
    e_b  = y_true - preds_df["y_pred_b"].values
    e_v3 = y_true - preds_df["y_pred_v3"].values
    dm_stat, p_val = diebold_mariano(e_b, e_v3)
    significant = p_val < 0.05
    better = "V3" if dm_stat < 0 else "B"
    print(f"\n  DM test (B vs V3): stat={dm_stat:.3f}, p={p_val:.4f} "
          f"({'significativo' if significant else 'NO significativo'})")
    print(f"  Mejor modelo segun DM: {better}")

    # --- Guardar ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    preds_sw.to_csv(OUTPUT_DIR / "xgb_v3_predictions.csv")
    print(f"[OK] Predicciones guardadas")

    meta = {
        "features_v3": features_v3,
        "n_features_b": len(features_b),
        "n_features_v3": len(features_v3),
        "hyperparams_v3": {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
        },
        "wfv_folds": len(folds_df),
        "B_MAPE_mean":  round(mape_b, 3),
        "V3_MAPE_mean": round(mape_v3, 3),
        "delta_mape":   round(mape_v3 - mape_b, 3),
        "V3_improved":  bool(mape_v3 < mape_b),
        "switch_B_MAPE":     round(mape_sw_b, 3),
        "switch_V3_MAPE":    round(mape_sw_v3, 3),
        "switch_original_MAPE": ORIGINAL_SWITCH_MAPE,
        "switch_V3_delta":   round(mape_sw_v3 - ORIGINAL_SWITCH_MAPE, 3),
        "switch_V3_improved": bool(mape_sw_v3 < ORIGINAL_SWITCH_MAPE),
        "dm_stat":      dm_stat,
        "dm_p_value":   p_val,
        "dm_significant": significant,
        "dm_better_model": better,
    }

    meta_path = OUTPUT_DIR / "xgb_v3_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[OK] Meta guardado: {meta_path}")

    # --- Plot ---
    print("\n[4/4] Generando plot...")
    make_plot(
        preds_sw, mape_b, mape_v3, mape_sw_b, mape_sw_v3,
        OUTPUT_DIR / "xgb_v3_plot.png"
    )

    # --- Resumen final ---
    print("\n" + "=" * 65)
    print("RESUMEN XGBoost V3")
    print("=" * 65)
    print(f"  XGBoost-B WFV:         {mape_b:.2f}% MAPE")
    print(f"  XGBoost-V3 WFV:        {mape_v3:.2f}% MAPE  [{mape_v3-mape_b:+.2f}pp]")
    print(f"  Switch B (ref):        {mape_sw_b:.2f}% MAPE")
    print(f"  Switch V3:             {mape_sw_v3:.2f}% MAPE  [{mape_sw_v3-ORIGINAL_SWITCH_MAPE:+.2f}pp vs original]")
    print(f"  DM test:               p={p_val:.4f} ({better} mejor, {'SIG' if significant else 'n.s.'})")
    print("\n[DONE] Step 07c completo.")


if __name__ == "__main__":
    main()
