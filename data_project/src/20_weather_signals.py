"""
Step 20 — Weather Signals via Open-Meteo Historical Archive API
API pública, sin clave, sin coste. Datos horarios agregados a semanal.

Ciudades representativas de Europa (ponderadas por población):
  Madrid, París, Berlín, Roma, Ámsterdam, Varsovia, Bucarest

Variables descargadas:
  temperature_2m_mean — temperatura media a 2m (°C)
  relative_humidity_2m_mean — humedad relativa media (%)

Período: 2012-01-01 a 2020-12-31 (cubre el dataset 2014-2019 con margen)

Output: data/external/weather_weekly_europe.csv
        data/external/weather_correlation.csv (CCF vs R03)

Run: python src/20_weather_signals.py
"""

import sys
import json
import time
from pathlib import Path

import requests
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    EXTERNAL_DIR, MAIN_DATASET, OPENMETEO_ARCHIVE_URL, WEATHER_CITIES
)

START_DATE = "2012-01-01"
END_DATE = "2020-12-31"
VARIABLES = "temperature_2m_mean,relative_humidity_2m_mean"
RETRY_DELAY = 3  # segundos entre reintentos


def fetch_city_weather(city: dict, start: str, end: str) -> pd.DataFrame | None:
    """
    Descarga datos diarios de temperatura y humedad para una ciudad vía Open-Meteo.
    Devuelve DataFrame con índice de fecha o None si falla.
    """
    params = {
        "latitude":  city["lat"],
        "longitude": city["lon"],
        "start_date": start,
        "end_date":   end,
        "daily":      VARIABLES,
        "timezone":   "UTC",
    }

    for attempt in range(3):
        try:
            r = requests.get(OPENMETEO_ARCHIVE_URL, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()

            if "daily" not in data:
                print(f"    [WARN] Sin datos 'daily' para {city['name']}")
                return None

            df = pd.DataFrame(data["daily"])
            df["time"] = pd.to_datetime(df["time"])
            df = df.set_index("time").sort_index()
            df.columns = [c.replace("_mean", "") for c in df.columns]
            return df

        except requests.exceptions.RequestException as e:
            print(f"    [WARN] Intento {attempt+1}/3 fallido para {city['name']}: {e}")
            time.sleep(RETRY_DELAY * (attempt + 1))

    print(f"    [ERROR] No se pudo descargar {city['name']}")
    return None


def compute_european_average(city_data: dict) -> pd.DataFrame:
    """
    Calcula la media europea ponderada por población de temperatura y humedad.
    city_data: {city_name: (df, weight)}
    """
    total_weight = sum(w for _, (_, w) in city_data.items())
    temp_weighted = None
    hum_weighted = None

    for city_name, (df, weight) in city_data.items():
        norm_w = weight / total_weight
        if "temperature_2m" in df.columns:
            t = df["temperature_2m"] * norm_w
            temp_weighted = t if temp_weighted is None else temp_weighted.add(t, fill_value=0)
        if "relative_humidity_2m" in df.columns:
            h = df["relative_humidity_2m"] * norm_w
            hum_weighted = h if hum_weighted is None else hum_weighted.add(h, fill_value=0)

    result = pd.DataFrame()
    if temp_weighted is not None:
        result["temp_europe_daily"] = temp_weighted
    if hum_weighted is not None:
        result["humidity_europe_daily"] = hum_weighted

    return result


def resample_to_sunday_week(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega datos diarios a semana-domingo (W-SUN), alineado con el dataset principal."""
    return df.resample("W-SUN").mean()


def compute_ccf(series_a: pd.Series, series_b: pd.Series, max_lag: int = 16) -> dict:
    """Cross-correlation para lags 0..max_lag. Devuelve lag óptimo y r."""
    aligned = pd.concat([series_a, series_b], axis=1).dropna()
    if len(aligned) < 30:
        return {"lag": 0, "r": float("nan"), "note": "insufficient data"}

    col_a, col_b = aligned.columns[0], aligned.columns[1]
    best_r, best_lag = -np.inf, 0
    all_lags = {}

    for lag in range(0, max_lag + 1):
        shifted = aligned[col_b].shift(lag)
        valid = pd.concat([aligned[col_a], shifted], axis=1).dropna()
        if len(valid) < 20:
            continue
        r = valid.corr().iloc[0, 1]
        all_lags[lag] = round(float(r), 4)
        if r > best_r:
            best_r, best_lag = r, lag

    return {"lag": best_lag, "r": round(float(best_r), 4), "all_lags": all_lags}


def main():
    print("=" * 60)
    print("STEP 20: WEATHER SIGNALS (OPEN-METEO)")
    print("=" * 60)

    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

    # --- Descargar datos por ciudad ---
    print(f"\n[1/4] Descargando datos meteorológicos para {len(WEATHER_CITIES)} ciudades...")
    city_data = {}

    for city in WEATHER_CITIES:
        print(f"  {city['name']} (lat={city['lat']}, lon={city['lon']})...")
        df = fetch_city_weather(city, START_DATE, END_DATE)
        if df is not None:
            city_data[city["name"]] = (df, city["weight"])
            print(f"    [OK] {len(df)} días")
        else:
            print(f"    [SKIP] {city['name']} omitida")

    if not city_data:
        print("[ERROR] No se descargaron datos de ninguna ciudad. Revisa la conexión.")
        return

    # --- Media europea ponderada ---
    print(f"\n[2/4] Calculando media europea ponderada ({len(city_data)} ciudades)...")
    eu_daily = compute_european_average(city_data)
    print(f"  Días disponibles: {len(eu_daily)}")
    print(f"  Rango: {eu_daily.index.min().date()} — {eu_daily.index.max().date()}")

    # --- Agregar a semana-domingo ---
    print("\n[3/4] Agregando a semana-domingo (W-SUN)...")
    eu_weekly = resample_to_sunday_week(eu_daily)
    print(f"  Semanas disponibles: {len(eu_weekly)}")

    out_path = EXTERNAL_DIR / "weather_weekly_europe.csv"
    eu_weekly.to_csv(out_path)
    print(f"[OK] Guardado: {out_path}")

    # Guardar también datos por ciudad para auditoría
    for city_name, (df_city, _) in city_data.items():
        city_weekly = resample_to_sunday_week(df_city)
        city_path = EXTERNAL_DIR / f"weather_{city_name.lower()}_weekly.csv"
        city_weekly.to_csv(city_path)

    # --- CCF contra R03 ---
    print("\n[4/4] Calculando correlación con R03...")
    if not MAIN_DATASET.exists():
        print(f"[WARN] Dataset principal no encontrado: {MAIN_DATASET}")
        print("  Ejecuta run_pipeline.py primero.")
        return

    target = pd.read_csv(MAIN_DATASET, parse_dates=["week_date"], index_col="week_date")["R03"]

    ccf_results = {}
    for col in eu_weekly.columns:
        ccf = compute_ccf(target, eu_weekly[col], max_lag=12)
        ccf_results[col] = ccf
        print(f"  {col}: r={ccf['r']:.3f} @ lag={ccf['lag']}w")

    ccf_df = pd.DataFrame([
        {"feature": k, "best_lag": v["lag"], "r": v["r"]}
        for k, v in ccf_results.items()
    ]).sort_values("r", ascending=False)

    ccf_path = EXTERNAL_DIR / "weather_correlation.csv"
    ccf_df.to_csv(ccf_path, index=False)
    print(f"[OK] Correlaciones guardadas: {ccf_path}")
    print(ccf_df.to_string(index=False))

    # Imprimir los lags óptimos para usar en integrate_v2
    print("\n  Lags óptimos para usar en 05b_integrate_v2.py:")
    for col, res in ccf_results.items():
        print(f"    {col}: lag={res['lag']}w (r={res['r']:.3f})")

    print("\n[DONE] Step 20 completo.")


if __name__ == "__main__":
    main()
