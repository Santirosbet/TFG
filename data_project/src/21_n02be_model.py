"""
Step 21 — N02BE (Paracetamol) Demand Forecasting

Aplica el mismo framework XGBoost del TFG a la categoria N02BE (paracetamol).
Hipotesis: N02BE tiene el mismo patron estacional que R03 (peak semana 52)
y deberia responder a la misma señal australiana con lag de ~20 semanas
(mas corto que R03 porque paracetamol se usa agudamente en fiebre, no de forma cronica).

Comparacion de targets:
  - R03 (broncodilatadores) : peak semana 52, uso cronico, lag AU = 26w
  - N02BE (paracetamol)     : peak semana 52, uso agudo,   lag AU = 20w

Features del modelo N02BE:
  - N02BE_lag1, N02BE_rolling4_mean (autoregresivas)
  - flu_au_lag20, flu_au_lag22, flu_au_lag24 (señal australiana, lag mas corto)
  - flu_eu_positives (señal europea contemporanea)
  - week_sin, week_cos (estacionalidad ciclica)
  - R03_lag1 (correlacion cruzada: R03 y N02BE co-mueven)

Outputs:
  output/n02be_meta.json        — metricas WFV + comparacion con R03
  output/n02be_predictions.csv  — predicciones out-of-sample
  output/n02be_plot.png         — visual

Run: python src/21_n02be_model.py
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
    SWITCHING_RULE_SUMMER_WEEKS, RANDOM_SEED
)

MIN_TRAIN   = 104
STEP        = 4
TEST_WINDOW = 4
LAG_AU_N02BE = 20   # lag optimo para paracetamol (CCF: r=0.344 a lag=20w)

BLUE   = "#1f4e79"
ORANGE = "#ff6b35"
GREEN  = "#70ad47"
PURPLE = "#7030a0"


def mape(y_true, y_pred):
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def diebold_mariano(e1, e2):
    d = np.abs(e1) - np.abs(e2)
    dm_stat, p_value = stats.ttest_1samp(d, 0)
    return float(dm_stat), float(p_value)


def build_model():
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


def build_n02be_dataset(base_df, n02be_series, flu_au_raw, flu_eu_series):
    """
    Construye dataset con N02BE como target.
    """
    df = pd.DataFrame(index=base_df.index)

    # Target
    n02be_aligned = n02be_series.reindex(df.index, method="nearest")
    df["N02BE"] = n02be_aligned

    # AR features de N02BE
    df["N02BE_lag1"]         = df["N02BE"].shift(1)
    df["N02BE_lag4_avg"]     = df["N02BE"].shift(1).rolling(4).mean()
    df["N02BE_rolling8_mean"]= df["N02BE"].shift(1).rolling(8).mean()
    df["N02BE_rolling4_std"] = df["N02BE"].shift(1).rolling(4).std()

    # Señal australiana a multiples lags (lag optimo para paracetamol = 20w)
    flu_au_aligned = flu_au_raw.reindex(df.index, method="nearest")
    df["flu_au_lag18"] = flu_au_aligned.shift(18)
    df["flu_au_lag20"] = flu_au_aligned.shift(20)
    df["flu_au_lag22"] = flu_au_aligned.shift(22)
    df["flu_au_positives"] = flu_au_aligned   # señal contemporanea

    # FluNet europeo
    df["flu_eu_positives"] = base_df["flu_eu_positives"].reindex(df.index, method="nearest")

    # Codificacion ciclica de semana
    week = df.index.isocalendar().week.astype(float)
    df["week_sin"] = np.sin(2 * np.pi * week / 52.18)
    df["week_cos"] = np.cos(2 * np.pi * week / 52.18)

    # R03 lag1 como feature cruzada (co-movimiento)
    df["R03_lag1"] = base_df["R03"].shift(1).reindex(df.index, method="nearest")

    df = df.dropna()
    return df


def run_wfv_n02be(df, features_a, features_b):
    """
    Model A: solo AR (lags N02BE propios)
    Model B: AR + señal australiana + cruzada
    """
    n = len(df)
    target = "N02BE"
    fold_records = []
    all_preds = []

    train_end = MIN_TRAIN
    fold_idx  = 0

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

        m_a = build_model(); m_a.fit(X_tr_a, y_tr)
        m_b = build_model(); m_b.fit(X_tr_b, y_tr)

        p_a = m_a.predict(X_te_a)
        p_b = m_b.predict(X_te_b)
        y_arr = y_te.values

        fold_records.append({
            "fold":        fold_idx + 1,
            "train_weeks": train_end,
            "test_start":  dates[0].strftime("%Y-%m-%d"),
            "test_end":    dates[-1].strftime("%Y-%m-%d"),
            "A_MAPE":      mape(y_arr, p_a),
            "B_MAPE":      mape(y_arr, p_b),
            "A_MAE":       mean_absolute_error(y_arr, p_a),
            "B_MAE":       mean_absolute_error(y_arr, p_b),
        })

        for i, dt in enumerate(dates):
            all_preds.append({
                "date":      dt,
                "y_true":    y_arr[i],
                "y_pred_a":  float(p_a[i]),
                "y_pred_b":  float(p_b[i]),
                "fold":      fold_idx + 1,
            })

        train_end += STEP
        fold_idx  += 1

    return pd.DataFrame(fold_records), pd.DataFrame(all_preds).set_index("date")


def apply_switching_n02be(preds_df, hist_mean):
    preds = preds_df.copy()
    week  = preds.index.isocalendar().week.astype(int)
    preds["y_switch"] = preds["y_pred_b"].copy()
    off_mask = week.isin(SWITCHING_RULE_SUMMER_WEEKS)
    for dt, row in preds[off_mask].iterrows():
        w = dt.isocalendar().week
        if w in hist_mean.index:
            preds.loc[dt, "y_switch"] = hist_mean[w]
    return preds


def make_plot(preds, mape_a, mape_b, mape_sw, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(14, 9),
                             gridspec_kw={"height_ratios": [3, 1]})

    ax1 = axes[0]
    ax1.plot(preds.index, preds["y_true"],   color=BLUE,   lw=2,   label="Actual N02BE (paracetamol)")
    ax1.plot(preds.index, preds["y_pred_a"], color=ORANGE, lw=1.5, linestyle="--", alpha=0.7,
             label=f"Model A — AR only  MAPE={mape_a:.1f}%")
    ax1.plot(preds.index, preds["y_pred_b"], color=GREEN,  lw=1.5,
             label=f"Model B — AR + AU flu  MAPE={mape_b:.1f}%")
    if "y_switch" in preds:
        ax1.plot(preds.index, preds["y_switch"], color=PURPLE, lw=2,
                 label=f"Switch rule  MAPE={mape_sw:.1f}%")
    ax1.set_title("N02BE (Paracetamol) — Forecasting with Australian Flu Signal", fontsize=13, fontweight="bold")
    ax1.set_ylabel("N02BE Units Sold (weekly)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    e_a = np.abs(preds["y_true"] - preds["y_pred_a"])
    e_b = np.abs(preds["y_true"] - preds["y_pred_b"])
    ax2.fill_between(preds.index, e_a, alpha=0.5, color=ORANGE, label=f"Error A (mean={e_a.mean():.1f})")
    ax2.fill_between(preds.index, e_b, alpha=0.5, color=GREEN,  label=f"Error B (mean={e_b.mean():.1f})")
    ax2.set_ylabel("Absolute Error")
    ax2.set_xlabel("Date")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Plot guardado: {out_path}")


def main():
    print("=" * 65)
    print("STEP 21: N02BE (PARACETAMOL) FORECASTING MODEL")
    print("=" * 65)

    # --- Cargar datos ---
    base = pd.read_csv(MAIN_DATASET, parse_dates=["week_date"], index_col="week_date")
    base = base.sort_index()

    sales_path = RAW_DIR / "salesweekly.csv"
    if not sales_path.exists():
        print(f"[ERROR] {sales_path} no encontrado")
        return
    sales = pd.read_csv(sales_path, parse_dates=["datum"], index_col="datum")
    sales.index.name = "week_date"
    n02be = sales["N02BE"].rename("N02BE").resample("W-SUN").mean()
    print(f"[OK] N02BE: {len(n02be)} semanas, media={n02be.mean():.1f}, std={n02be.std():.1f}")

    flu_path = PROCESSED_DIR / "flunet_australia.csv"
    if flu_path.exists():
        flu_raw = pd.read_csv(flu_path, parse_dates=["iso_date"]).set_index("iso_date")
        flu_au  = flu_raw["INF_ALL"].resample("W-SUN").sum()
    else:
        flu_au = base["flu_au_positives"]
    print(f"[OK] FluNet AU bruto: {len(flu_au)} semanas")

    # --- Feature engineering ---
    print("\n[1/4] Construyendo dataset N02BE...")
    df = build_n02be_dataset(base, n02be, flu_au, base["flu_eu_positives"])
    print(f"[OK] Dataset N02BE: {df.shape}")
    print(f"  R03 media = {base['R03'].mean():.1f} | N02BE media = {n02be.mean():.1f}")

    features_a = [
        "N02BE_lag1", "N02BE_lag4_avg",
        "N02BE_rolling8_mean",
        "week_sin", "week_cos",
    ]

    features_b = [
        "N02BE_lag1", "N02BE_lag4_avg",
        "N02BE_rolling8_mean", "N02BE_rolling4_std",
        "flu_au_lag18", "flu_au_lag20", "flu_au_lag22",
        "flu_au_positives", "flu_eu_positives",
        "week_sin", "week_cos",
        "R03_lag1",
    ]

    features_a = [f for f in features_a if f in df.columns]
    features_b = [f for f in features_b if f in df.columns]

    # --- WFV ---
    print("\n[2/4] Walk-Forward Validation N02BE...")
    folds_df, preds_df = run_wfv_n02be(df, features_a, features_b)
    mape_a = folds_df["A_MAPE"].mean()
    mape_b = folds_df["B_MAPE"].mean()
    b_wins = (folds_df["B_MAPE"] < folds_df["A_MAPE"]).mean() * 100

    print(f"  Model A (AR only):          MAPE = {mape_a:.3f}%")
    print(f"  Model B (AR + AU flu):      MAPE = {mape_b:.3f}%")
    print(f"  B wins pct: {b_wins:.1f}% of folds")
    print(f"  Delta (B vs A): {mape_b - mape_a:+.3f}pp")

    # --- Switching ---
    print("\n[3/4] Switching rule N02BE...")
    hist_base = df.iloc[:MIN_TRAIN].copy()
    hist_base["week"] = hist_base.index.isocalendar().week.astype(int)
    hist_mean = hist_base.groupby("week")["N02BE"].mean()

    preds_sw = apply_switching_n02be(preds_df, hist_mean)
    y_true = preds_sw["y_true"].values
    mape_sw = mape(y_true, preds_sw["y_switch"].values)
    print(f"  Switch rule MAPE: {mape_sw:.3f}%")

    # --- DM test A vs B ---
    e_a = y_true - preds_df["y_pred_a"].values
    e_b = y_true - preds_df["y_pred_b"].values
    dm_stat, p_val = diebold_mariano(e_a, e_b)
    significant = p_val < 0.05
    better = "B (AU flu)" if dm_stat < 0 else "A (AR only)"
    print(f"  DM test A vs B: stat={dm_stat:.3f}, p={p_val:.4f} "
          f"({'SIG' if significant else 'n.s.'}), mejor: {better}")

    # --- Guardar ---
    preds_sw.to_csv(OUTPUT_DIR / "n02be_predictions.csv")

    meta = {
        "target": "N02BE",
        "description": "Paracetamol/Acetaminophen (N02BE) demand forecasting",
        "lag_au_optimal": LAG_AU_N02BE,
        "n02be_mean": round(float(n02be.mean()), 2),
        "n02be_std":  round(float(n02be.std()), 2),
        "r03_mean":   round(float(base["R03"].mean()), 2),
        "wfv_folds":  len(folds_df),
        "A_MAPE_mean":  round(mape_a, 3),
        "B_MAPE_mean":  round(mape_b, 3),
        "B_delta_pp":   round(mape_b - mape_a, 3),
        "B_wins_pct":   round(b_wins, 1),
        "switch_MAPE":  round(mape_sw, 3),
        "dm_stat":      dm_stat,
        "dm_p_value":   p_val,
        "dm_significant": significant,
        "dm_better":    better,
        "comparison_r03": {
            "R03_B_MAPE":    48.626,
            "N02BE_B_MAPE":  round(mape_b, 3),
            "interpretation": "Comparar para validar que la hipotesis se generaliza"
        }
    }
    with open(OUTPUT_DIR / "n02be_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[OK] Meta guardado: output/n02be_meta.json")

    # --- Plot ---
    print("\n[4/4] Generando plot...")
    make_plot(preds_sw, mape_a, mape_b, mape_sw,
              OUTPUT_DIR / "n02be_plot.png")

    # --- Resumen ---
    print("\n" + "=" * 65)
    print("RESUMEN N02BE")
    print("=" * 65)
    print(f"  Model A (AR only):     {mape_a:.2f}% MAPE")
    print(f"  Model B (+ AU flu):    {mape_b:.2f}% MAPE  [{mape_b-mape_a:+.2f}pp]")
    print(f"  Switching rule:        {mape_sw:.2f}% MAPE")
    print(f"  DM test:               p={p_val:.4f} ({better}, {'SIG' if significant else 'n.s.'})")
    print(f"\n  Referencia R03-B WFV:  48.63% MAPE")
    interpretation = "MEJOR" if mape_b < 48.626 else "PEOR"
    print(f"  N02BE-B vs R03-B:      {interpretation}")
    print("\n[DONE] Step 21 completo.")


if __name__ == "__main__":
    main()
