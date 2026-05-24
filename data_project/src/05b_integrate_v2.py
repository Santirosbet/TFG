"""
Step 05b — Integrate External Signals into Dataset v2
Añade Google Trends y señales meteorológicas al dataset original.

IMPORTANTE: El integrated_dataset.csv original NO se modifica.
Este script produce integrated_dataset_v2.csv como archivo nuevo.

Nuevas features:
  trends_gripe_es_lag{N}    — Google Trends España "gripe" con lag óptimo
  trends_resfriado_es_lag{N} — Google Trends España "resfriado" con lag óptimo
  temp_europe_lag{N}        — Temperatura media europea con lag óptimo
  humidity_europe_lag{N}    — Humedad relativa europea con lag óptimo

Los lags se determinan automáticamente leyendo los archivos CCF generados
por los steps 19 y 20. Si no existen, usa lags por defecto documentados.

Run: python src/05b_integrate_v2.py
"""

import sys
import json
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    MAIN_DATASET, ENHANCED_DATASET, EXTERNAL_DIR, PROCESSED_DIR, LAG_AU_FLU
)

# Lags por defecto si no hay archivos CCF (basados en literatura y sentido común)
DEFAULT_LAGS = {
    "trends_gripe_es":    2,   # búsquedas preceden a compras ~2 semanas
    "trends_resfriado_es": 1,
    "trends_paracetamol_es": 1,
    "temp_europe":        4,   # temperatura fría → demanda ~4 semanas después
    "humidity_europe":    3,
}


def load_optimal_lags() -> dict:
    """
    Lee los archivos CCF de steps 19 y 20.
    Devuelve dict {feature_base_name: lag_weeks}.
    """
    lags = dict(DEFAULT_LAGS)

    trends_ccf = EXTERNAL_DIR / "google_trends_correlation.csv"
    if trends_ccf.exists():
        df = pd.read_csv(trends_ccf)
        for _, row in df.iterrows():
            feature = row["feature"]  # ej: "trends_gripe_es"
            base = feature.replace("trends_", "").replace("_es", "").replace("_eu", "")
            key = f"trends_{base}_es" if "_es" in feature else f"trends_{base}_eu"
            lags[key] = int(row["best_lag"])
        print(f"[OK] Lags Google Trends cargados desde {trends_ccf}")
    else:
        print(f"[WARN] {trends_ccf} no encontrado — usando lags por defecto para Trends")

    weather_ccf = EXTERNAL_DIR / "weather_correlation.csv"
    if weather_ccf.exists():
        df = pd.read_csv(weather_ccf)
        for _, row in df.iterrows():
            if "temp" in row["feature"]:
                lags["temp_europe"] = int(row["best_lag"])
            elif "humidity" in row["feature"]:
                lags["humidity_europe"] = int(row["best_lag"])
        print(f"[OK] Lags meteorológicos cargados desde {weather_ccf}")
    else:
        print(f"[WARN] {weather_ccf} no encontrado — usando lags por defecto para weather")

    return lags


def load_trends() -> pd.DataFrame | None:
    """Carga Google Trends semanal si existe."""
    trends_path = EXTERNAL_DIR / "google_trends_weekly.csv"
    if not trends_path.exists():
        print(f"[WARN] {trends_path} no encontrado — omitiendo señales de búsqueda")
        print("       Ejecuta src/19_google_trends.py primero")
        return None
    df = pd.read_csv(trends_path, parse_dates=["date"], index_col="date")
    df = df.sort_index()
    print(f"[OK] Google Trends: {df.shape[1]} series, {len(df)} semanas")
    return df


def load_weather() -> pd.DataFrame | None:
    """Carga señales meteorológicas semanales si existen."""
    weather_path = EXTERNAL_DIR / "weather_weekly_europe.csv"
    if not weather_path.exists():
        print(f"[WARN] {weather_path} no encontrado — omitiendo señales meteorológicas")
        print("       Ejecuta src/20_weather_signals.py primero")
        return None
    df = pd.read_csv(weather_path, parse_dates=["time"], index_col="time")
    df = df.sort_index()
    print(f"[OK] Weather: {df.shape[1]} series, {len(df)} semanas")
    return df


def apply_lag(series: pd.Series, lag: int) -> pd.Series:
    """Desplaza la serie lag semanas hacia adelante (el predictor lidera al target)."""
    return series.shift(lag)


def main():
    print("=" * 60)
    print("STEP 05b: INTEGRATE EXTERNAL SIGNALS — DATASET V2")
    print("=" * 60)

    # --- Cargar dataset original ---
    if not MAIN_DATASET.exists():
        print(f"[ERROR] Dataset principal no encontrado: {MAIN_DATASET}")
        print("  Ejecuta run_pipeline.py (steps 1-5) primero.")
        return

    base = pd.read_csv(MAIN_DATASET, parse_dates=["week_date"], index_col="week_date")
    base = base.sort_index()
    print(f"[OK] Dataset base: {base.shape} — {base.index.min().date()} a {base.index.max().date()}")
    print(f"     Columnas originales: {list(base.columns)}")

    # --- Lags óptimos ---
    lags = load_optimal_lags()
    print(f"\n[INFO] Lags a aplicar: {lags}")

    v2 = base.copy()
    new_features = []

    # --- Google Trends ---
    print("\n[1/3] Añadiendo señales Google Trends...")
    trends = load_trends()
    if trends is not None:
        target_terms = {
            "trends_gripe_es":     "trends_gripe_es",
            "trends_resfriado_es": "trends_resfriado_es",
            "trends_paracetamol_es": "trends_paracetamol_es",
        }
        for feature_key, col_pattern in target_terms.items():
            # Buscar la columna que coincida (el nombre puede variar levemente)
            matching = [c for c in trends.columns if col_pattern in c]
            if not matching:
                print(f"  [SKIP] '{col_pattern}' no encontrado en trends")
                continue
            col = matching[0]
            lag = lags.get(feature_key, DEFAULT_LAGS.get(feature_key, 2))
            feature_name = f"{feature_key}_lag{lag}"
            v2[feature_name] = apply_lag(trends[col].reindex(v2.index, method="nearest"), lag)
            new_features.append(feature_name)
            print(f"  [+] {feature_name} (lag={lag}w)")

    # --- Señales meteorológicas ---
    print("\n[2/3] Añadiendo señales meteorológicas...")
    weather = load_weather()
    if weather is not None:
        weather_targets = {
            "temp_europe":     "temp_europe_daily",
            "humidity_europe": "humidity_europe_daily",
        }
        for feature_key, col_name in weather_targets.items():
            if col_name not in weather.columns:
                alt = [c for c in weather.columns if feature_key.split("_")[0] in c]
                col_name = alt[0] if alt else None
            if col_name is None:
                print(f"  [SKIP] '{feature_key}' no encontrado en weather")
                continue
            lag = lags.get(feature_key, DEFAULT_LAGS.get(feature_key, 4))
            feature_name = f"{feature_key}_lag{lag}"
            v2[feature_name] = apply_lag(weather[col_name].reindex(v2.index, method="nearest"), lag)
            new_features.append(feature_name)
            print(f"  [+] {feature_name} (lag={lag}w)")

    # --- Limpieza de NaN generados por los lags ---
    print("\n[3/3] Limpiando NaN...")
    before = len(v2)
    # Solo dropna en las nuevas columnas si existen; preservar las filas del original
    if new_features:
        v2[new_features] = v2[new_features].bfill().ffill()
    after = len(v2)
    print(f"  Filas antes: {before} | después: {after}")

    # --- Guardar ---
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    v2.to_csv(ENHANCED_DATASET)
    print(f"\n[OK] Dataset v2 guardado: {ENHANCED_DATASET}")
    print(f"     Shape: {v2.shape}")
    print(f"     Columnas nuevas: {new_features}")
    print(f"     Total columnas: {len(v2.columns)}")

    # --- Resumen de correlación con R03 ---
    print("\n--- CORRELACIÓN DE NUEVAS FEATURES CON R03 ---")
    for feat in new_features:
        if feat in v2.columns:
            r = v2["R03"].corr(v2[feat])
            print(f"  {feat}: r = {r:.3f}")

    print("\n[DONE] Step 05b completo.")
    print(f"  Dataset v2 listo para usar en 07b_model_lightgbm.py y 16b_stacking_ensemble.py")


if __name__ == "__main__":
    main()
