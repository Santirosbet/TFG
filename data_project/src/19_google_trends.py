"""
Step 19 — Google Trends Signal
Descarga tendencias semanales de búsquedas relacionadas con enfermedades
respiratorias en España/Europa como señal líder adicional.

Términos: gripe, resfriado, paracetamol, bronquitis, tos, inhalador
Geografía: ES (España) + EU (media europea)
Período: 2012-2020 (cubrir el dataset 2014-2019 con margen)

Output: data/external/google_trends_weekly.csv
        data/external/google_trends_correlation.csv (CCF vs R03)

Run: python src/19_google_trends.py
"""

import sys
import time
import json
from pathlib import Path

import pandas as pd
import numpy as np

# Añadir raíz al path para importar config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    EXTERNAL_DIR, MAIN_DATASET, GOOGLE_TRENDS_DELAY, RANDOM_SEED
)

TERMS_ES = ["gripe", "resfriado", "paracetamol", "bronquitis", "tos", "inhalador"]
TERMS_EU = ["flu", "cold", "paracetamol", "bronchitis", "cough", "inhaler"]

START_DATE = "2012-01-01"
END_DATE = "2020-12-31"


def download_trends(terms: list, geo: str, delay: float) -> pd.DataFrame:
    """Descarga Google Trends para una lista de términos y geografía."""
    try:
        from pytrends.request import TrendReq
    except ImportError:
        print("[ERROR] pytrends no instalado. Ejecuta: pip install pytrends")
        return pd.DataFrame()

    pytrends = TrendReq(hl="en-US", tz=0, timeout=(10, 25), retries=2, backoff_factor=0.5)
    frames = {}

    for term in terms:
        print(f"  Descargando '{term}' ({geo})...")
        try:
            pytrends.build_payload(
                [term],
                cat=0,
                timeframe=f"{START_DATE} {END_DATE}",
                geo=geo,
                gprop=""
            )
            df = pytrends.interest_over_time()
            if df.empty:
                print(f"    [WARN] Sin datos para '{term}' en {geo}")
            else:
                frames[term] = df[term]
                print(f"    [OK] {len(df)} semanas")
            time.sleep(delay)
        except Exception as e:
            print(f"    [ERROR] '{term}' ({geo}): {e}")
            time.sleep(delay * 2)  # backoff extra en caso de bloqueo

    if not frames:
        return pd.DataFrame()

    result = pd.DataFrame(frames)
    result.index = pd.to_datetime(result.index)
    result.index.name = "date"
    return result


def resample_to_sunday_week(df: pd.DataFrame) -> pd.DataFrame:
    """Alinea las fechas a domingo (W-SUN) para coincidir con el dataset principal."""
    return df.resample("W-SUN").mean()


def compute_ccf(series_a: pd.Series, series_b: pd.Series, max_lag: int = 20) -> dict:
    """
    Calcula la cross-correlation entre series_a y series_b para lags 0..max_lag.
    Devuelve el lag con correlación máxima.
    """
    results = {}
    a = series_a.dropna()
    b = series_b.dropna()
    aligned = pd.concat([a, b], axis=1).dropna()
    if len(aligned) < 30:
        return {"lag": 0, "r": np.nan, "note": "insufficient data"}

    col_a, col_b = aligned.columns
    best_r, best_lag = -np.inf, 0
    for lag in range(0, max_lag + 1):
        shifted = aligned[col_b].shift(lag)
        valid = pd.concat([aligned[col_a], shifted], axis=1).dropna()
        if len(valid) < 20:
            continue
        r = valid.corr().iloc[0, 1]
        results[lag] = r
        if r > best_r:
            best_r, best_lag = r, lag

    return {"lag": best_lag, "r": round(float(best_r), 4), "all_lags": results}


def main():
    print("=" * 60)
    print("STEP 19: GOOGLE TRENDS SIGNAL")
    print("=" * 60)

    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

    # --- Descargar términos España ---
    print("\n[1/4] Descargando términos ES (España)...")
    df_es = download_trends(TERMS_ES, geo="ES", delay=GOOGLE_TRENDS_DELAY)

    # --- Descargar términos Europa (sin geo = global; 'EU' no siempre funciona en pytrends) ---
    print("\n[2/4] Descargando términos EU (Europa)...")
    df_eu = download_trends(TERMS_EU, geo="ES-MD", delay=GOOGLE_TRENDS_DELAY)
    # Nota: pytrends con geo="EU" puede fallar; usamos ES-MD como proxy europeo continental

    # --- Combinar y alinear a semana-domingo ---
    frames_to_merge = []

    if not df_es.empty:
        df_es = resample_to_sunday_week(df_es)
        df_es.columns = [f"trends_{c}_es" for c in df_es.columns]
        frames_to_merge.append(df_es)

    if not df_eu.empty:
        df_eu = resample_to_sunday_week(df_eu)
        df_eu.columns = [f"trends_{c}_eu" for c in df_eu.columns]
        frames_to_merge.append(df_eu)

    if not frames_to_merge:
        print("\n[WARN] No se descargaron datos. Saliendo.")
        print("  Posibles causas: Google bloqueó las requests o pytrends no instalado.")
        print("  Reintenta más tarde o aumenta GOOGLE_TRENDS_DELAY en .env")
        return

    trends_combined = pd.concat(frames_to_merge, axis=1)
    trends_combined = trends_combined.sort_index()

    # Normalizar a escala 0-100 (Google ya devuelve 0-100, pero por si hay variación)
    for col in trends_combined.columns:
        col_max = trends_combined[col].max()
        if col_max > 0:
            trends_combined[col] = trends_combined[col] / col_max * 100

    out_path = EXTERNAL_DIR / "google_trends_weekly.csv"
    trends_combined.to_csv(out_path)
    print(f"\n[OK] Guardado: {out_path}")
    print(f"     Shape: {trends_combined.shape}")
    print(f"     Rango: {trends_combined.index.min().date()} — {trends_combined.index.max().date()}")

    # --- CCF contra R03 ---
    print("\n[3/4] Calculando correlación con R03...")
    if not MAIN_DATASET.exists():
        print(f"[WARN] Dataset principal no encontrado: {MAIN_DATASET}")
        print("  Ejecuta run_pipeline.py primero para generar integrated_dataset.csv")
        return

    target = pd.read_csv(MAIN_DATASET, parse_dates=["week_date"], index_col="week_date")["R03"]

    ccf_results = {}
    for col in trends_combined.columns:
        aligned = pd.concat([target, trends_combined[col]], axis=1).dropna()
        if len(aligned) < 30:
            continue
        ccf = compute_ccf(aligned.iloc[:, 0], aligned.iloc[:, 1], max_lag=8)
        ccf_results[col] = ccf
        print(f"  {col}: r={ccf['r']:.3f} @ lag={ccf['lag']}w")

    # Guardar resultados de correlación
    ccf_df = pd.DataFrame([
        {"feature": k, "best_lag": v["lag"], "r": v["r"]}
        for k, v in ccf_results.items()
    ]).sort_values("r", ascending=False)

    ccf_path = EXTERNAL_DIR / "google_trends_correlation.csv"
    ccf_df.to_csv(ccf_path, index=False)
    print(f"[OK] Correlaciones guardadas: {ccf_path}")

    # --- Selección de mejores términos ---
    print("\n[4/4] Mejores términos por correlación con R03:")
    print(ccf_df.to_string(index=False))

    best_terms = ccf_df[ccf_df["r"] > 0.20]["feature"].tolist()
    print(f"\n  Términos con r > 0.20: {best_terms}")
    print("  Usar estos en 05b_integrate_v2.py")

    print("\n[DONE] Step 19 completo.")


if __name__ == "__main__":
    main()
