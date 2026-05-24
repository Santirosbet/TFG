"""
Step 29 — Enriched Feature Model (temperature + RSV + trends + seasonal)
=========================================================================
Extends the base XGBoost model with all available external signals:
  - Temperature & humidity (7 EU cities via Open-Meteo)
  - RSV activity (WHO FluNet)
  - Google Trends (flu-related searches, ES + EU)
  - Seasonal sine/cosine encodings
  - More autoregressive lags of R03 and R06

Walk-forward validation (same 48-fold protocol) and season-by-season
backtest (train -> predict full European winter season).

Outputs:
  output/enriched_meta.json       — metrics + feature importances
  output/enriched_predictions.csv — test set predictions
  output/enriched_backtest.csv    — season-by-season validation table
  output/enriched_features.csv    — full enriched dataset

Run: python src/29_enriched_model.py
"""

import os, json, warnings, io, requests
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
warnings.filterwarnings("ignore")
import sys
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")

ROOT     = os.path.join(os.path.dirname(__file__), "..")
PROC     = os.path.join(ROOT, "data", "processed")
EXT      = os.path.join(ROOT, "data", "external")
OUT      = os.path.join(ROOT, "output")
os.makedirs(OUT, exist_ok=True)

# -────────────────────────────────────────────────────────────────────────────
# 1. HELPERS
# -────────────────────────────────────────────────────────────────────────────
def mape(y_true, y_pred):
    mask = y_true > 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def lag(series, n):
    return series.shift(n)

def rolling_avg(series, n):
    return series.shift(1).rolling(n, min_periods=1).mean()

# -────────────────────────────────────────────────────────────────────────────
# 2. LOAD BASE DATA
# -────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("STEP 29: ENRICHED FEATURE MODEL")
print("=" * 65)

base = pd.read_csv(os.path.join(PROC, "integrated_dataset_v2.csv"))
base["week_date"] = pd.to_datetime(base["week_date"])
base = base.sort_values("week_date").reset_index(drop=True)
print(f"[BASE] {len(base)} rows  {base['week_date'].min().date()} -> {base['week_date'].max().date()}")

# -────────────────────────────────────────────────────────────────────────────
# 3. LOAD TEMPERATURE  (weekly average of 7 EU cities)
# -────────────────────────────────────────────────────────────────────────────
weather = pd.read_csv(os.path.join(EXT, "weather_weekly_europe.csv"))
weather["time"] = pd.to_datetime(weather["time"])
weather = weather.rename(columns={"time": "week_date",
                                   "temp_europe_daily":     "temp_eu",
                                   "humidity_europe_daily": "humidity_eu"})
weather["week_date"] = weather["week_date"] + pd.Timedelta(days=6)   # align to week-end Sunday
weather = weather.resample("W-SUN", on="week_date").mean().reset_index()
print(f"[WEATHER] {len(weather)} weekly rows")

# -────────────────────────────────────────────────────────────────────────────
# 4. LOAD GOOGLE TRENDS  (keep strongest signals: gripe_es, flu_eu)
# -────────────────────────────────────────────────────────────────────────────
trends = pd.read_csv(os.path.join(EXT, "google_trends_weekly.csv"))
trends["date"] = pd.to_datetime(trends["date"])
trends = trends.rename(columns={"date": "week_date"})
# Keep the two most correlated signals + composite
trends["trends_flu_composite"] = (
    trends[["trends_gripe_es", "trends_flu_eu", "trends_tos_es"]]
    .fillna(0).mean(axis=1)
)
trends = trends[["week_date", "trends_gripe_es", "trends_flu_eu",
                  "trends_flu_composite"]].ffill()
print(f"[TRENDS] {len(trends)} weekly rows")

# -────────────────────────────────────────────────────────────────────────────
# 5. FETCH RSV FROM WHO FLUNET API  (for 2012-2020)
# -────────────────────────────────────────────────────────────────────────────
RSV_CACHE = os.path.join(EXT, "rsv_europe_weekly.csv")
if os.path.exists(RSV_CACHE):
    rsv = pd.read_csv(RSV_CACHE, parse_dates=["week_date"])
    print(f"[RSV] loaded from cache  ({len(rsv)} rows)")
else:
    print("[RSV] fetching from WHO FluNet API ...")
    try:
        r = requests.get(
            "https://xmart-api-public.who.int/FLUMART/VIW_FNT",
            params={"$format": "csv",
                    "$filter": "WHOREGION eq 'EUR' and ISO_YEAR ge 2012 and ISO_YEAR le 2020"},
            timeout=60,
        )
        r.raise_for_status()
        raw = pd.read_csv(io.StringIO(r.text), low_memory=False)
        raw["RSV"] = pd.to_numeric(raw.get("RSV", 0), errors="coerce").fillna(0)
        raw["week_date"] = pd.to_datetime(
            raw["ISO_YEAR"].astype(int).astype(str)
            + raw["ISO_WEEK"].astype(int).astype(str).str.zfill(2) + "1",
            format="%G%V%u", errors="coerce",
        )
        raw = raw.dropna(subset=["week_date"])
        rsv = raw.groupby("week_date")["RSV"].sum().reset_index()
        rsv.to_csv(RSV_CACHE, index=False)
        print(f"[RSV] fetched & cached  ({len(rsv)} rows)")
    except Exception as e:
        print(f"[RSV] API unavailable ({e}) — using zeros")
        rsv = pd.DataFrame({"week_date": base["week_date"], "RSV": 0})

# -────────────────────────────────────────────────────────────────────────────
# 6. MERGE ALL SOURCES
# -────────────────────────────────────────────────────────────────────────────
df = base.copy()

def nearest_merge(df, right, on="week_date", tol_days=7):
    """Merge on nearest date within tolerance."""
    right = right.rename(columns={on: "_rdate"})
    merged = pd.merge_asof(
        df.sort_values(on),
        right.sort_values("_rdate"),
        left_on=on, right_on="_rdate",
        direction="nearest",
        tolerance=pd.Timedelta(days=tol_days),
    ).drop(columns=["_rdate"])
    return merged.sort_values(on).reset_index(drop=True)

df = nearest_merge(df, weather)
df = nearest_merge(df, trends)
df = nearest_merge(df, rsv)
print(f"[MERGED] {len(df)} rows, {len(df.columns)} cols")

# -────────────────────────────────────────────────────────────────────────────
# 7. FEATURE ENGINEERING
# -────────────────────────────────────────────────────────────────────────────
d = df.copy()

# -─ Autoregressive R03  (core demand history) -──────────────────────────────
for k in [1, 4, 8, 26, 52]:
    d[f"R03_lag{k}"] = lag(d["R03"], k)
d["R03_roll4"]  = rolling_avg(d["R03"], 4)
d["R03_roll8"]  = rolling_avg(d["R03"], 8)

# -─ R06 as demand context -──────────────────────────────────────────────────
d["R06_lag1"] = lag(d["R06"], 1)

# -─ Australian flu signal (the core lead-lag feature) -─────────────────────
for k in [26, 27, 28]:
    d[f"flu_au_lag{k}"] = lag(d["flu_au_positives"], k)
d["flu_au_roll4_lag26"] = rolling_avg(d["flu_au_positives"], 4).shift(26)
d["flu_au_roll8_lag26"] = rolling_avg(d["flu_au_positives"], 8).shift(26)

# -─ European flu (1-2 week lag only — concurrent is leakage) -───────────────
d["flu_eu_lag1"] = lag(d["flu_eu_positives"], 1)
d["flu_eu_lag2"] = lag(d["flu_eu_positives"], 2)

# -─ Temperature signals (min 2-week lag: reporting delay) -──────────────────
for k in [2, 4, 8]:
    d[f"temp_eu_lag{k}"] = lag(d["temp_eu"], k)
d["temp_eu_roll4"] = rolling_avg(d["temp_eu"], 4)

# -─ Google Trends (1-2 week lag only) -──────────────────────────────────────
d["trends_gripe_es_lag1"] = lag(d["trends_gripe_es"], 1)
d["trends_flu_eu_lag1"]   = lag(d["trends_flu_eu"], 1)
d["trends_composite_lag1"] = lag(d["trends_flu_composite"], 1)

# -─ RSV (lag 1+: last week data available; NO lag 0 — same-week leakage) -─────
if d["RSV"].sum() > 0:
    for k in [1, 2, 4, 8]:
        d[f"RSV_lag{k}"] = lag(d["RSV"], k)
    d["RSV_roll4"] = rolling_avg(d["RSV"], 4)

# -─ Seasonal encoding (week-of-year) -──────────────────────────────────────
d["week_num"]  = d["week_date"].dt.isocalendar().week.astype(float)
d["week_sin"]  = np.sin(2 * np.pi * d["week_num"] / 52)
d["week_cos"]  = np.cos(2 * np.pi * d["week_num"] / 52)
d["month_sin"] = np.sin(2 * np.pi * d["week_date"].dt.month / 12)
d["month_cos"] = np.cos(2 * np.pi * d["week_date"].dt.month / 12)

# -─ Year trend -─────────────────────────────────────────────────────────────
d["year_num"] = d["week_date"].dt.year - d["week_date"].dt.year.min()

# Save enriched dataset
d.to_csv(os.path.join(OUT, "enriched_features.csv"), index=False)
print(f"[FEATURES] {len(d.columns)} total columns engineered")

# -────────────────────────────────────────────────────────────────────────────
# 8. EXPLICIT FEATURE LIST  (curated: ~22 features, no contemporaneous leakage)
# -────────────────────────────────────────────────────────────────────────────
# All features have minimum 1-week lag (or 4+ for RSV/temp) to avoid leakage.
# Keeping <25 features avoids overfitting on 196 training samples.
RSV_COLS = ([f"RSV_lag{k}" for k in [1, 2, 4, 8]] + ["RSV_roll4"]
            if d["RSV"].sum() > 0 else [])

valid_features = [
    # Demand autoregression
    "R03_lag1", "R03_lag4", "R03_lag8", "R03_lag26", "R03_lag52",
    "R03_roll4", "R03_roll8",
    "R06_lag1",
    # Lead-lag Australian signal
    "flu_au_lag26", "flu_au_lag27", "flu_au_lag28",
    "flu_au_roll4_lag26", "flu_au_roll8_lag26",
    # European flu context
    "flu_eu_lag1", "flu_eu_lag2",
    # Climate
    "temp_eu_lag2", "temp_eu_lag4", "temp_eu_lag8", "temp_eu_roll4",
    # Google Trends
    "trends_gripe_es_lag1", "trends_flu_eu_lag1", "trends_composite_lag1",
    # Seasonality
    "week_sin", "week_cos", "month_sin", "month_cos",
    # Year trend
    "year_num",
] + RSV_COLS

# Keep only columns that actually exist in df
valid_features = [f for f in valid_features if f in d.columns]
print(f"[FEATURES] {len(valid_features)} curated features (no contemporaneous leakage)")

# -────────────────────────────────────────────────────────────────────────────
# 9. TRAIN / TEST SPLIT  (same 80/20 as original)
# -────────────────────────────────────────────────────────────────────────────
d_clean = d[valid_features + ["R03", "week_date"]].dropna()
print(f"[DATA] After dropna: {len(d_clean)} rows  "
      f"({d_clean['week_date'].min().date()} -> {d_clean['week_date'].max().date()})")

# Use same proportional split
split_idx = int(len(d_clean) * 0.80)
train = d_clean.iloc[:split_idx]
test  = d_clean.iloc[split_idx:]
X_tr  = train[valid_features].values
y_tr  = train["R03"].values
X_te  = test[valid_features].values
y_te  = test["R03"].values

print(f"[SPLIT] train={len(train)}  test={len(test)}")

# -────────────────────────────────────────────────────────────────────────────
# 10. XGBOOST WITH HYPERPARAMETER TUNING
# -────────────────────────────────────────────────────────────────────────────
from sklearn.model_selection import TimeSeriesSplit

best_params = {
    "n_estimators":     300,
    "max_depth":        3,
    "learning_rate":    0.05,
    "subsample":        0.75,
    "colsample_bytree": 0.65,
    "min_child_weight": 5,
    "reg_alpha":        0.5,
    "reg_lambda":       2.0,
    "random_state":     42,
    "n_jobs":           -1,
}

# Quick grid search over key params using TimeSeriesSplit CV
tscv  = TimeSeriesSplit(n_splits=5)
grid  = {
    "max_depth":     [3, 4],
    "learning_rate": [0.03, 0.05],
    "n_estimators":  [200, 300, 500],
}

print("[TUNING] Grid search (5-fold TimeSeriesCV) ...")
best_cv_mape = float("inf")
for depth in grid["max_depth"]:
    for lr in grid["learning_rate"]:
        for n_est in grid["n_estimators"]:
            fold_mapes = []
            for tr_idx, va_idx in tscv.split(X_tr):
                m = xgb.XGBRegressor(
                    n_estimators=n_est, max_depth=depth, learning_rate=lr,
                    subsample=0.75, colsample_bytree=0.65, min_child_weight=5,
                    reg_alpha=0.5, reg_lambda=2.0, random_state=42,
                    n_jobs=-1, verbosity=0,
                )
                m.fit(X_tr[tr_idx], y_tr[tr_idx], verbose=False)
                p    = m.predict(X_tr[va_idx])
                fold_mapes.append(mape(y_tr[va_idx], p))
            cv_mape = np.mean(fold_mapes)
            if cv_mape < best_cv_mape:
                best_cv_mape = cv_mape
                best_params.update({"max_depth": depth, "learning_rate": lr,
                                     "n_estimators": n_est})

print(f"[TUNING] Best CV MAPE: {best_cv_mape:.2f}% | params: "
      f"depth={best_params['max_depth']} lr={best_params['learning_rate']} "
      f"n={best_params['n_estimators']}")

# Final model
model = xgb.XGBRegressor(**best_params, verbosity=0)
model.fit(X_tr, y_tr, verbose=False)

y_pred = model.predict(X_te)
y_pred = np.maximum(y_pred, 0)

test_mape = mape(y_te, y_pred)
test_mae  = float(mean_absolute_error(y_te, y_pred))
test_rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))

print(f"\n{'='*50}")
print(f"  ENRICHED MODEL MAPE : {test_mape:.2f}%")
print(f"  MAE                 : {test_mae:.2f} units/week")
print(f"  RMSE                : {test_rmse:.2f} units/week")
print(f"  Original XGBoost    : 44.16%")
print(f"  Switching Rule      : 35.78%")
improvement = 35.78 - test_mape
print(f"  Improvement vs best : {improvement:+.2f} pp")
print(f"{'='*50}\n")

# Feature importances
feat_imp = pd.Series(model.feature_importances_, index=valid_features)
feat_imp = feat_imp.sort_values(ascending=False)
print("Top 15 features:")
for i, (f, v) in enumerate(feat_imp.head(15).items()):
    print(f"  {i+1:2d}. {f:35s} {v:.4f}")

# Save test predictions
pred_df = test[["week_date", "R03"]].copy()
pred_df["predicted"] = y_pred
pred_df.to_csv(os.path.join(OUT, "enriched_predictions.csv"), index=False)

# -────────────────────────────────────────────────────────────────────────────
# 11. WALK-FORWARD VALIDATION  (48 folds, same protocol as original)
# -────────────────────────────────────────────────────────────────────────────
print("\n[WFV] Running 48-fold walk-forward validation ...")
N = len(d_clean)
min_train = int(N * 0.5)
step = 1
wfv_rows = []

for fold, test_end in enumerate(range(min_train + 1, N + 1, step)):
    tr = d_clean.iloc[:test_end - 1]
    te = d_clean.iloc[test_end - 1 : test_end]
    if len(tr) < min_train or te.empty:
        continue
    m = xgb.XGBRegressor(**best_params, verbosity=0)
    m.fit(tr[valid_features].values, tr["R03"].values, verbose=False)
    pred = float(np.maximum(m.predict(te[valid_features].values)[0], 0))
    actual = float(te["R03"].iloc[0])
    err = abs(pred - actual) / max(actual, 1) * 100
    wfv_rows.append({"fold": fold, "date": te["week_date"].iloc[0],
                     "actual": actual, "predicted": pred, "mape": err})

wfv_df = pd.DataFrame(wfv_rows)
wfv_mape = wfv_df["mape"].mean()
print(f"[WFV] Mean MAPE across {len(wfv_df)} folds: {wfv_mape:.2f}%")

# -────────────────────────────────────────────────────────────────────────────
# 12. SEASON-BY-SEASON BACKTEST
# -────────────────────────────────────────────────────────────────────────────
print("\n[BACKTEST] Season-by-season validation ...")

# European winter seasons in our dataset (weeks 44-13 of following year)
seasons = [
    {"name": "2015-16", "train_end": "2015-06-01",
     "pred_start": "2015-11-01", "pred_end": "2016-03-31"},
    {"name": "2016-17", "train_end": "2016-06-01",
     "pred_start": "2016-11-01", "pred_end": "2017-03-31"},
    {"name": "2017-18", "train_end": "2017-06-01",
     "pred_start": "2017-11-01", "pred_end": "2018-03-31"},
    {"name": "2018-19", "train_end": "2018-06-01",
     "pred_start": "2018-11-01", "pred_end": "2019-03-31"},
]

backtest_rows = []
for s in seasons:
    train_mask = d_clean["week_date"] < s["train_end"]
    test_mask  = (d_clean["week_date"] >= s["pred_start"]) & \
                 (d_clean["week_date"] <= s["pred_end"])

    s_train = d_clean[train_mask]
    s_test  = d_clean[test_mask]

    if len(s_train) < 52 or s_test.empty:
        print(f"  {s['name']}: not enough data (train={len(s_train)} test={len(s_test)})")
        continue

    m = xgb.XGBRegressor(**best_params, verbosity=0)
    m.fit(s_train[valid_features].values, s_train["R03"].values, verbose=False)

    preds  = np.maximum(m.predict(s_test[valid_features].values), 0)
    actual = s_test["R03"].values

    season_mape     = mape(actual, preds)
    peak_actual     = float(actual.max())
    peak_pred       = float(preds.max())
    peak_week_actual = s_test["week_date"].iloc[actual.argmax()].strftime("%Y-%m-%d")
    peak_week_pred   = s_test["week_date"].iloc[preds.argmax()].strftime("%Y-%m-%d")
    total_actual    = float(actual.sum())
    total_pred      = float(preds.sum())
    total_err_pct   = (total_pred - total_actual) / max(total_actual, 1) * 100

    row = {
        "Temporada": s["name"],
        "Semanas test": len(s_test),
        "MAPE temporada (%)": round(season_mape, 1),
        "Demanda total real": round(total_actual, 0),
        "Demanda total predicha": round(total_pred, 0),
        "Error total (%)": round(total_err_pct, 1),
        "Pico real (u/sem)": round(peak_actual, 1),
        "Pico predicho (u/sem)": round(peak_pred, 1),
        "Fecha pico real": peak_week_actual,
        "Fecha pico predicha": peak_week_pred,
        "Semanas diferencia pico": abs(
            (pd.Timestamp(peak_week_pred) - pd.Timestamp(peak_week_actual)).days // 7
        ),
    }
    backtest_rows.append(row)
    print(f"  {s['name']}: MAPE={season_mape:.1f}%  "
          f"Peak real={peak_actual:.0f}  pred={peak_pred:.0f}  "
          f"pico_err={abs(peak_actual-peak_pred)/max(peak_actual,1)*100:.1f}%")

backtest_df = pd.DataFrame(backtest_rows)
backtest_df.to_csv(os.path.join(OUT, "enriched_backtest.csv"), index=False)
print("\nSeason-by-season backtest table:")
print(backtest_df.to_string(index=False))

# -────────────────────────────────────────────────────────────────────────────
# 13. SAVE META
# -────────────────────────────────────────────────────────────────────────────
meta_out = {
    "model": "XGBoost Enriched (temperature + RSV + Google Trends + seasonal)",
    "n_features": len(valid_features),
    "features": list(feat_imp.head(20).index),
    "feature_importances": {f: round(float(v), 5) for f, v in feat_imp.head(20).items()},
    "test_mape_pct": round(test_mape, 2),
    "test_mae": round(test_mae, 2),
    "test_rmse": round(test_rmse, 2),
    "wfv_mean_mape_pct": round(wfv_mape, 2),
    "wfv_n_folds": len(wfv_df),
    "baseline_switching_mape": 35.78,
    "baseline_xgboost_mape": 44.16,
    "improvement_vs_switching_pp": round(35.78 - test_mape, 2),
    "improvement_vs_xgboost_pp": round(44.16 - test_mape, 2),
    "best_params": best_params,
    "train_rows": len(train),
    "test_rows": len(test),
}
with open(os.path.join(OUT, "enriched_meta.json"), "w") as f:
    json.dump(meta_out, f, indent=2)

print(f"\n[DONE] All outputs saved to output/")
print(f"  enriched_meta.json")
print(f"  enriched_predictions.csv")
print(f"  enriched_backtest.csv")
print(f"  enriched_features.csv")
