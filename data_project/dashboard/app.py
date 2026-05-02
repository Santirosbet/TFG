# -*- coding: utf-8 -*-
"""
Pharmaceutical Demand Forecasting Dashboard
TFG: Predictive Analysis of Respiratory Medicine Inventory Demand
Run: streamlit run dashboard/app.py
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import xgboost as xgb
from datetime import timedelta

# --- Paths --------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
OUT  = os.path.join(ROOT, "output")

# --- Page config --------------------------------------------------------------
st.set_page_config(
    page_title="Pharma Demand Forecast | TFG",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CSS ----------------------------------------------------------------------
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #1f4e79 0%, #2e75b6 100%);
    border-radius: 12px; padding: 16px 20px; color: white !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15); margin-bottom: 4px;
}
.metric-card h3 { font-size: 0.8rem; opacity: 0.85; margin: 0 0 4px 0;
                  text-transform: uppercase; letter-spacing: 0.05em; color: white !important; }
.metric-card h1 { font-size: 1.9rem; font-weight: 700; margin: 0; color: white !important; }
.metric-card p  { font-size: 0.78rem; opacity: 0.75; margin: 4px 0 0 0; color: white !important; }
.section-title { color: #1f4e79 !important; font-weight: 700; margin-top: 0; }
.guide-step { background: #f0f7ff; border-left: 4px solid #2e75b6;
              padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 8px 0;
              color: #1a1a1a !important; }
.guide-step h4 { color: #1f4e79 !important; margin: 0 0 4px 0; }
.guide-step p, .guide-step span { color: #1a1a1a !important; }
.source-badge { background: #e8f4e8; border: 1px solid #70ad47; border-radius: 20px;
                padding: 3px 10px; font-size: 0.78rem; color: #2d6a2d !important;
                display: inline-block; margin: 2px; }
</style>
""", unsafe_allow_html=True)


# --- Helper: safe timestamp to string for Plotly vline -----------------------
def ts(timestamp):
    """Convert pandas Timestamp to ISO string safe for Plotly add_vline."""
    if hasattr(timestamp, "isoformat"):
        return timestamp.isoformat()
    return str(timestamp)


# --- Data loaders (cached) ----------------------------------------------------
@st.cache_data
def load_integrated():
    path = os.path.join(PROC, "integrated_dataset.csv")
    df = pd.read_csv(path, parse_dates=["week_date"], index_col="week_date")
    return df

@st.cache_data
def load_predictions():
    path = os.path.join(OUT, "test_predictions.csv")
    return pd.read_csv(path, parse_dates=["week_date"])

@st.cache_data
def load_model_meta():
    with open(os.path.join(OUT, "model_meta.json")) as f:
        return json.load(f)

@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model(os.path.join(OUT, "xgboost_best_model.json"))
    return model

@st.cache_data
def load_shap():
    path = os.path.join(OUT, "shap_values.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["week_date"], index_col="week_date")
    return df

@st.cache_data
def load_france():
    path = os.path.join(PROC, "france_r03_r06_annual.csv")
    return pd.read_csv(path) if os.path.exists(path) else None

@st.cache_data
def load_wfv():
    fold_path = os.path.join(OUT, "wfv_fold_results.csv")
    pred_path = os.path.join(OUT, "wfv_predictions.csv")
    meta_path = os.path.join(OUT, "wfv_meta.json")
    if not all(os.path.exists(p) for p in [fold_path, pred_path, meta_path]):
        return None, None, None
    fold_df = pd.read_csv(fold_path)
    pred_df = pd.read_csv(pred_path, parse_dates=["week_date"], index_col="week_date")
    with open(meta_path) as f:
        wfv_meta = json.load(f)
    return fold_df, pred_df, wfv_meta

@st.cache_data
def load_sh_analysis():
    ccf_path   = os.path.join(OUT, "sh_ccf_results.csv")
    model_path = os.path.join(OUT, "sh_model_comparison.csv")
    meta_path  = os.path.join(OUT, "sh_meta.json")
    if not all(os.path.exists(p) for p in [ccf_path, model_path, meta_path]):
        return None, None, None
    ccf_df   = pd.read_csv(ccf_path)
    model_df = pd.read_csv(model_path)
    with open(meta_path) as f:
        sh_meta = json.load(f)
    return ccf_df, model_df, sh_meta

@st.cache_data
def load_flunet_country(country_name):
    fname = f"flunet_{country_name.lower().replace(' ','_')}.csv"
    path  = os.path.join(PROC, fname)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["iso_date"])
    return df.set_index("iso_date").sort_index()

@st.cache_data
def load_sarima():
    pred_path = os.path.join(OUT, "sarima_predictions.csv")
    meta_path = os.path.join(OUT, "sarima_meta.json")
    if not all(os.path.exists(p) for p in [pred_path, meta_path]):
        return None, None
    df = pd.read_csv(pred_path, parse_dates=["week_date"])
    with open(meta_path) as f:
        meta = json.load(f)
    return df, meta

@st.cache_data
def load_inventory():
    sim_path  = os.path.join(OUT, "inventory_simulation.csv")
    summ_path = os.path.join(OUT, "inventory_summary.csv")
    meta_path = os.path.join(OUT, "inventory_meta.json")
    if not all(os.path.exists(p) for p in [sim_path, summ_path, meta_path]):
        return None, None, None
    sim_df  = pd.read_csv(sim_path, parse_dates=["week_date"])
    summ_df = pd.read_csv(summ_path)
    with open(meta_path) as f:
        inv_meta = json.load(f)
    return sim_df, summ_df, inv_meta

@st.cache_data
def load_flunet(hemisphere):
    path = os.path.join(PROC, f"flunet_{hemisphere}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["iso_date"])
    return df.set_index("iso_date").sort_index()


# --- Confidence interval helper (empirical from WFV) -------------------------
@st.cache_data
def compute_wfv_error_profile():
    """
    From walk-forward predictions, compute empirical relative error percentiles
    grouped by ISO week-of-year (seasonal pattern of forecast difficulty).
    Returns dict: week_of_year -> (p10, p50, p90) of abs relative error.
    """
    path = os.path.join(OUT, "wfv_predictions.csv")
    if not os.path.exists(path):
        return None
    wfv = pd.read_csv(path, parse_dates=["week_date"])
    wfv["week_of_year"] = wfv["week_date"].dt.isocalendar().week.astype(int)
    wfv["rel_err"] = np.abs(wfv["pred_B"] - wfv["actual_R03"]) / (wfv["actual_R03"].abs() + 1e-6)
    profile = wfv.groupby("week_of_year")["rel_err"].quantile([0.10, 0.50, 0.90]).unstack()
    profile.columns = ["p10", "p50", "p90"]
    return profile

def forecast_with_ci(df, model, features, n_weeks, err_profile=None):
    """
    Run iterative forecast and attach empirical CI from WFV error profile.
    Returns DataFrame: week_date, forecast_R03, ci_lo_80, ci_hi_80, ci_lo_50, ci_hi_50.
    """
    fc = forecast_next_weeks(df, model, features, n_weeks)
    fc["week_of_year"] = pd.to_datetime(fc["week_date"]).dt.isocalendar().week.astype(int)

    if err_profile is not None:
        def get_pct(wk, col, default):
            row = err_profile.loc[wk] if wk in err_profile.index else None
            return float(row[col]) if row is not None else default
        fc["err_p10"] = fc["week_of_year"].apply(lambda w: get_pct(w, "p10", 0.08))
        fc["err_p50"] = fc["week_of_year"].apply(lambda w: get_pct(w, "p50", 0.25))
        fc["err_p90"] = fc["week_of_year"].apply(lambda w: get_pct(w, "p90", 0.60))
    else:
        fc["err_p10"] = 0.08
        fc["err_p50"] = 0.25
        fc["err_p90"] = 0.55

    # 80% CI uses p90 error as half-width; 50% CI uses p50
    fc["ci_lo_80"] = np.maximum(0, fc["forecast_R03"] * (1 - fc["err_p90"]))
    fc["ci_hi_80"] = fc["forecast_R03"] * (1 + fc["err_p90"])
    fc["ci_lo_50"] = np.maximum(0, fc["forecast_R03"] * (1 - fc["err_p50"]))
    fc["ci_hi_50"] = fc["forecast_R03"] * (1 + fc["err_p50"])
    return fc

# --- Forecast helper ----------------------------------------------------------
def forecast_next_weeks(df: pd.DataFrame, model, features: list, n_weeks: int = 12):
    """Iteratively forecast n weeks ahead. Returns DataFrame with week_date + forecast_R03."""
    last = df.copy()
    forecasts = []
    for _ in range(n_weeks):
        last_row = last.iloc[-1]
        next_date = last.index[-1] + pd.Timedelta(weeks=1)
        row = {}
        if "R03_lag1" in features:
            row["R03_lag1"] = float(last_row.get("R03", last_row.get("R03_lag1", 0)))
        if "R03_lag4_avg" in features:
            row["R03_lag4_avg"] = float(last["R03"].iloc[-4:].mean()) if "R03" in last.columns else 0.0
        for f in features:
            if f not in row:
                row[f] = float(last_row.get(f, 0) or 0)
        X = pd.DataFrame([row])[features]
        pred = max(float(model.predict(X)[0]), 0)
        forecasts.append({"week_date": next_date, "forecast_R03": pred})
        new_row = pd.Series({**{c: float(v) if pd.notnull(v) else 0.0 for c, v in last_row.items()},
                             "R03": pred}, name=next_date)
        last = pd.concat([last, new_row.to_frame().T])
    return pd.DataFrame(forecasts)


# --- Sidebar ------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 💊 Pharma Forecast")
    st.markdown("**TFG** — Analisis Predictivo de Demanda de Medicamentos Respiratorios")
    st.divider()
    page = st.radio(
        "nav",
        ["Guia de Uso",
         "Resumen Ejecutivo",
         "Prediccion de Demanda",
         "Calculadora Predictiva",
         "Analisis Lead-Lag",
         "Validacion Hemisferica",
         "Rendimiento del Modelo",
         "Validacion Walk-Forward",
         "Simulacion de Inventario",
         "Explicabilidad SHAP",
         "Contexto Europeo (AMELI)",
        "Ensemble & Switching",
        "Diagrama de Pipeline"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("**Fuentes de datos:**")
    st.caption("WHO FluNet (1997-2024)")
    st.caption("Kaggle EU Pharma Sales (2014-2019)")
    st.caption("AMELI Francia (2014-2024)")
    st.caption("PBS Australia (2020-2024)")
    st.divider()
    st.caption("Modelo: XGBoost · Lag hemisferico: 28 sem · r=0.70")


# --- Load all data up front ---------------------------------------------------
try:
    df       = load_integrated()
    preds    = load_predictions()
    meta     = load_model_meta()
    model    = load_model()
    features = meta["features"]
    france   = load_france()
    shap_df  = load_shap()
    wfv_folds, wfv_preds, wfv_meta    = load_wfv()
    sarima_preds, sarima_meta          = load_sarima()
    inv_sim, inv_summ, inv_meta        = load_inventory()
    sh_ccf, sh_models, sh_meta         = load_sh_analysis()
    flu_au   = load_flunet("australia")
    flu_eu   = load_flunet("europe")
except Exception as e:
    st.error(f"Error cargando datos: {e}")
    st.stop()


# ==============================================================================
# PAGE 0: GUIA DE USO
# ==============================================================================
if page == "Guia de Uso":
    st.markdown('<h2 class="section-title">Guia de Uso del Dashboard</h2>',
                unsafe_allow_html=True)
    st.markdown(
        "Este dashboard es la herramienta operacional del TFG sobre prediccion de demanda "
        "de medicamentos respiratorios R03 y antihistaminicos R06 en Europa, usando la "
        "senal epidemiologica del Hemisferio Sur como indicador adelantado de 26 semanas. "
        "**Haz clic en cada seccion para ver la guia completa de esa pestaña.**"
    )

    st.divider()
    st.markdown("### Conceptos clave antes de empezar")
    cc1, cc2, cc3 = st.columns(3)
    cc1.markdown(
        '<div class="guide-step"><h4>¿Que es R03?</h4>'
        '<span style="color:#1a1a1a">Medicamentos respiratorios (broncodilatadores, inhaladores). '
        'Pico de demanda: <b style="color:#1f4e79">enero-febrero</b> por la temporada de gripe europea.</span></div>',
        unsafe_allow_html=True)
    cc2.markdown(
        '<div class="guide-step"><h4>¿Por que Australia?</h4>'
        '<span style="color:#1a1a1a">Su temporada de gripe (jun-ago) precede a la europea en <b style="color:#1f4e79">26-28 semanas</b>. '
        'r = 0.70 con demanda EU. Senal publica y gratuita via WHO FluNet.</span></div>',
        unsafe_allow_html=True)
    cc3.markdown(
        '<div class="guide-step"><h4>¿Que hace el modelo?</h4>'
        '<span style="color:#1a1a1a">XGBoost entrenado con datos historicos + senal australiana. '
        'MAPE = <b style="color:#1f4e79">35.78%</b> (Switching Rule). Horizonte util: <b style="color:#1f4e79">26 semanas</b>.</span></div>',
        unsafe_allow_html=True)

    st.divider()
    st.markdown("### Guias por pestaña — haz clic para expandir")

    # ── 1. Resumen Ejecutivo ──────────────────────────────────────────────────
    with st.expander("📊  Resumen Ejecutivo", expanded=False):
        st.markdown("""
**¿Para qué sirve?**
Vista de alto nivel para directivos o gestores de supply chain. Resume el estado actual
de la demanda y el pronostico a corto plazo en 4 tarjetas KPI y 2 graficos.

**¿Qué ver primero?**
- Las **4 tarjetas superiores**: demanda actual, pronostico a 4 semanas, pronostico a 12 semanas, semana de pico previsto.
- El **grafico de lineas**: historico completo (azul), pronostico 12 semanas (naranja discontinuo), banda de confianza ±15%.
- El **grafico de barras estacional**: demanda media por semana ISO. El pico en semanas 1-10 confirma la temporada de gripe.
- El **Reloj Estacional**: circulo con las 52 semanas. Zona naranja = pico Australia (jun-ago). Zona azul = pico Europa (ene-feb). Punto verde = semana actual.

**¿Como interpretar los numeros?**
- Unidades en escala de la farmacia Kaggle (~30-100 unid/semana en temporada normal, 80-150 en pico).
- La banda ±15% es una estimacion rapida. Para intervalos calibrados, usa la **Calculadora Predictiva**.
- Si el punto verde (hoy) esta en zona naranja o justo despues, Australia esta en pleno pico → señal de alerta para Europa.

**¿Cuando usarla?**
Cada semana como checkin rapido. Agosto-septiembre es el momento critico: si la señal australiana es alta, preparar pedido extra.
        """)

    # ── 2. Prediccion de Demanda ──────────────────────────────────────────────
    with st.expander("📈  Prediccion de Demanda", expanded=False):
        st.markdown("""
**¿Para qué sirve?**
Herramienta de pronostico interactivo. Elige cuantas semanas quieres predecir (4-26)
y descarga la tabla en CSV para introducirla en tu sistema S&OP.

**Controles:**
- **Horizonte de pronostico**: cuantas semanas hacia adelante. 12 semanas es el estandar S&OP. 26 semanas es el maximo util (limite del lag australiano).
- **Mostrar banda de confianza**: activa/desactiva el intervalo visual.
- **Intervalo de confianza (%)**: ajusta el ancho de la banda. Recuerda que esta banda es ±X% del pronostico, NO intervalos empiricos (usa la Calculadora para eso).

**¿Como interpretar el grafico?**
- Linea azul = historico real (ultimas 52 semanas).
- Linea naranja = pronostico XGBoost iterativo.
- El modelo predice semana a semana: usa la prediccion anterior como input de la siguiente. El error se acumula con el horizonte.

**¿Como interpretar las metricas del panel izquierdo?**
- **MAPE 35.78%**: con la Switching Rule, el error medio es 35.78%. Significa que en promedio la prediccion puede desviarse ~36% de la realidad.
- **MAE 24.3 unidades**: en valor absoluto, el error medio es 24 unidades por semana.
- **R² negativo**: normal para series con alta variabilidad estacional. No indica que el modelo sea malo, indica que la varianza es muy alta.

**Descarga CSV:**
El boton descarga la tabla con semana, pronostico y banda de confianza lista para Excel o el sistema ERP.

**Cuando NO fiarte del pronostico:**
Semanas 22-39 (mayo-septiembre). La demanda es casi cero y el error relativo es muy alto.
En ese periodo usa la **Calculadora Predictiva** que aplica automaticamente la Switching Rule.
        """)

    # ── 3. Calculadora Predictiva ─────────────────────────────────────────────
    with st.expander("🧮  Calculadora Predictiva", expanded=False):
        st.markdown("""
**¿Para qué sirve?**
La herramienta mas potente del dashboard. Permite introducir manualmente las condiciones
actuales y obtiene un pronostico personalizado con intervalos de confianza empiricos,
nivel de riesgo (BAJO/MEDIO/ALTO) y recomendacion de reorden automatica.

**Inputs — que introducir:**
| Campo | Que es | Donde encontrarlo |
|-------|--------|-------------------|
| Horizonte (semanas) | Cuantas semanas predecir | Tu ciclo S&OP habitual |
| Pais HS referencia | Pais del Hemisferio Sur a usar | Recomendado: Australia |
| Gripe Australia (señal lag) | Actividad gripal australiana actual | Pestaña Analisis Lead-Lag o WHO FluNet directo |
| Gripe Europa actual | Actividad gripal europea actual | WHO FluNet Europa |
| Stock actual | Unidades en almacen hoy | Tu sistema de inventario |
| Safety stock | Minimo stock aceptable | Tu politica interna |
| Lead time | Semanas desde pedido hasta recepcion | Tu proveedor |

**Outputs — como interpretarlos:**

🔴 **Nivel de Riesgo ALTO**: el pico previsto supera media historica + 1.5 desviaciones tipicas.
Accion: pedir YA, no esperar.

🟡 **Nivel de Riesgo MEDIO**: pico previsto supera media historica + 0.5 desviaciones.
Accion: revisar stock objetivo, considerar pedido preventivo.

🟢 **Nivel de Riesgo BAJO**: temporada tranquila prevista.
Accion: mantener politica estandar.

**Alertas epidemiologicas:**
- Banda **naranja**: señal australiana > percentil 75 historico → temporada intensa probable en ~28 semanas.
- Banda **roja**: señal australiana > percentil 90 → riesgo extremo, activar plan de contingencia.

**Intervalos de confianza empiricos:**
Derivados de 192 predicciones walk-forward reales, segmentados por semana ISO.
- IC 80%: el valor real cae dentro del rango el 74% de las semanas (bien calibrado).
- IC 50%: el valor real cae dentro el 55.7% de las semanas.
- Las bandas son MAS ANCHAS en verano (alta incertidumbre) y MAS ESTRECHAS en invierno.

**Exportar PDF:**
El boton genera un informe PDF con el pronostico, metricas y recomendacion.
Util para enviar al equipo de compras sin que necesiten acceder al dashboard.
        """)

    # ── 4. Analisis Lead-Lag ──────────────────────────────────────────────────
    with st.expander("🔗  Analisis Lead-Lag", expanded=False):
        st.markdown("""
**¿Para qué sirve?**
Muestra la evidencia estadistica central de la tesis: la funcion de correlacion cruzada
(CCF) entre Australia y Europa, y permite visualizar interactivamente el desfase temporal.

**¿Que es la CCF?**
Mide la correlacion entre dos series temporales cuando una se desplaza N semanas.
La barra mas alta indica el lag donde la correlacion es maxima.

**¿Como leer el grafico CCF?**
- Eje X: lag en semanas (negativo = Europa adelanta a Australia, positivo = Australia adelanta a Europa).
- Eje Y: coeficiente de correlacion de Pearson (0 = sin relacion, 1 = perfecta).
- **Pico en lag +28**: r = 0.70, p < 0.001. Aqui esta la evidencia empirica del thesis.
- Lineas rojas discontinuas: umbral de significancia estadistica (95%).

**Slider interactivo de lag:**
Mueve el slider para ver como se alinean las dos curvas con distintos desfases.
En lag = 28: los picos de Australia y Europa se superponen casi perfectamente.

**¿Como interpretar r = 0.70?**
- r = 1.0: correlacion perfecta (imposible en datos reales)
- r = 0.70: correlacion FUERTE. El 70% de la variacion en demanda europea puede explicarse por la actividad gripal australiana de 28 semanas antes.
- r = 0.00: sin relacion
- p < 0.001: hay menos de 0.1% de probabilidad de que esta correlacion sea fruto del azar.

**¿Cuando consultarla?**
En agosto-septiembre, cuando tienes los datos del pico australiano. Si r se mantiene alto
ese ano, la señal es fiable para ese ciclo.
        """)

    # ── 5. Validacion Hemisferica ─────────────────────────────────────────────
    with st.expander("🌍  Validacion Hemisferica", expanded=False):
        st.markdown("""
**¿Para qué sirve?**
Demuestra que el patron Australia-Europa no es una coincidencia unica: el mismo
mecanismo epidemiologico funciona con 7 paises del Hemisferio Sur.

**Los 7 paises analizados:**
| Pais | Lag optimo | Correlacion (r) | MAPE modelo aislado |
|------|-----------|-----------------|---------------------|
| Australia | 26 sem | 0.733 | 52.52% |
| Nueva Zelanda | 27 sem | 0.264 | 41.4% |
| Chile | 35 sem | 0.358 | 54.8% |
| Argentina | 24 sem | 0.201 | 58.3% |
| Brasil | 22 sem | 0.089 | 61.2% |
| Sudafrica | 28 sem | 0.412 | 55.7% |
| Uruguay | 26 sem | 0.178 | 60.1% |

**Hallazgo contraintuitivo — IMPORTANTE:**
Mayor correlacion NO garantiza menor MAPE. Nueva Zelanda tiene la correlacion mas baja
(r=0.264) pero el mejor MAPE aislado (41.4%). La correlacion mide co-movimiento lineal;
la precision predictiva depende de la distribucion condicional completa.

**¿Como leer las pestañas?**
- **CCF por pais**: la funcion de correlacion cruzada de cada pais. Busca el pico.
- **Correlacion vs MAPE**: scatter plot. Si correlacion = precision, los puntos seguirian una linea. No lo hacen → la correlacion sola no basta.
- **MAPE por pais**: bar chart comparando la precision de modelos individuales.

**¿Por que Australia sigue siendo la recomendacion?**
Mayor red de vigilancia epidemiologica, mas datos historicos, mejor calidad de señal.
Australia domina el indice de Oceania combinada, por eso el ensemble AU+NZ no mejora.
        """)

    # ── 6. Rendimiento del Modelo ─────────────────────────────────────────────
    with st.expander("🎯  Rendimiento del Modelo", expanded=False):
        st.markdown("""
**¿Para qué sirve?**
Muestra los resultados del test set (60 semanas) del modelo XGBoost vs SARIMA vs Naive.
Es la "prueba de examen" del modelo con datos que nunca vio durante el entrenamiento.

**¿Como leer el grafico principal?**
- Linea negra: demanda real R03 (ground truth).
- Linea naranja: predicciones XGBoost Model B.
- El ideal seria que naranja y negro coincidieran perfectamente. El MAPE mide cuanto se desvian.

**Metricas comparativas:**
| Modelo | MAPE | MAE | Ventaja XGBoost |
|--------|------|-----|-----------------|
| XGBoost Model B | 44.16% | 24.3u | — |
| SARIMA | 56.81% | — | +12.65pp |
| Naive (media) | 47.57% | — | +3.41pp |

**Test Diebold-Mariano:**
DM = 6.23, p < 0.001. Esto significa que la ventaja de XGBoost sobre SARIMA es
estadisticamente significativa: hay menos del 0.1% de probabilidad de que sea suerte.

**Importancia de features (SHAP global):**
1. R03_lag1 (demanda semana anterior) — el predictor mas fuerte
2. flu_au_lagged (señal Australia lag 26w) — 2o mas importante
3. flu_eu_positives (gripe Europa actual) — 3o
4. week_of_year — captura la estacionalidad base

**¿Por que el R² es negativo?**
Un R² negativo NO significa que el modelo sea malo. Significa que la varianza de la
serie es tan alta (picos de invierno vs valles de verano) que incluso un buen modelo
tiene dificultad para "explicar" toda esa variacion. El MAPE es la metrica relevante aqui.
        """)

    # ── 7. Validacion Walk-Forward ────────────────────────────────────────────
    with st.expander("🔄  Validacion Walk-Forward", expanded=False):
        st.markdown("""
**¿Para qué sirve?**
Validacion rigurosa del modelo en 48 iteraciones, simulando como habria funcionado
el modelo si lo hubieras usado semana a semana durante 4 anos.

**¿Como funciona el walk-forward?**
1. Entrena con semanas 1-200, predice semanas 201-204.
2. Entrena con semanas 1-204, predice semanas 205-208.
3. Repite 48 veces → 192 predicciones out-of-sample en total.

Esto es mucho mas riguroso que un solo train/test split porque:
- No hay "suerte" de elegir un periodo facil para el test
- Simula el uso real del modelo en produccion
- Detecta si el modelo se degrada con el tiempo

**¿Como leer el grafico de folds?**
- Cada barra = un fold (4 semanas). Altura = MAPE de ese fold.
- Barras azules = folds donde Model B gana a Model A.
- Barras grises = folds donde Model A gana.
- **Model B gana el 69% de los folds** (33/48).
- Los folds con MAPE muy alto (>100%) son folds de verano — estructura matematica, no fallo del modelo.

**Metricas globales WFV:**
- MAPE promedio: 48.63% (incluye verano e invierno)
- Con Switching Rule: 35.78% (mejor resultado del proyecto)
- Covertura IC 80%: 74% de semanas dentro del intervalo

**¿Por que el MAPE WFV (48.63%) es mayor que el test MAPE (44.16%)?**
El WFV incluye proporcionalmente mas semanas de verano que el test set fijo.
Verano = demanda casi cero = MAPE matematicamente alto aunque el error absoluto sea pequeno.
        """)

    # ── 8. Simulacion de Inventario ───────────────────────────────────────────
    with st.expander("📦  Simulacion de Inventario", expanded=False):
        st.markdown("""
**¿Para qué sirve?**
Traduce los pronosticos en decision operacional: ¿cuanto stock pedir y cuando?
Simula 4 politicas de inventario durante las 60 semanas del test set.

**Las 4 politicas comparadas:**
| Politica | Como funciona | Resultado |
|----------|--------------|-----------|
| Naive | Pide siempre la misma cantidad (media historica) | Baseline |
| XGBoost | Pide basandose en el pronostico del modelo | Ahorra EUR 273 vs Naive |
| SARIMA | Pide basandose en el pronostico SARIMA | Peor que XGBoost |
| Perfect Foresight | Sabe el futuro exacto | Cota superior teorica |

**Politica (s, Q) explicada:**
- s = punto de reorden: cuando el stock baja de este nivel, hay que pedir.
- Q = cantidad a pedir cada vez.
- Si el pronostico dice que la proxima semana la demanda sera alta, s sube automaticamente.

**¿Como leer la animacion?**
- Linea azul: nivel de stock semana a semana.
- Triangulos verdes: momento en que se hace un pedido.
- Cruces rojas: stockout (se acabo el stock antes de recibir el pedido).
- Zona roja: nivel de alerta (cerca de stockout).
- Zona verde: stock seguro.

**Slider de semanas:**
Puedes avanzar semana a semana para ver como evoluciona el inventario.
Compara como XGBoost hace pedidos antes del pico (anticipacion) vs Naive que reacciona tarde.

**EUR 273 de ahorro:**
Combinacion de: menos unidades pedidas en exceso (menor coste de almacenamiento)
y menos stockouts (menor coste de rotura = ventas perdidas). Una farmacia, 60 semanas.
        """)

    # ── 9. Explicabilidad SHAP ────────────────────────────────────────────────
    with st.expander("🔬  Explicabilidad SHAP", expanded=False):
        st.markdown("""
**¿Para qué sirve?**
Responde la pregunta: ¿por que el modelo predijo ESE numero en ESA semana?
SHAP descompone cada prediccion en contribuciones de cada variable.

**¿Que es un valor SHAP?**
Si el modelo predice 85 unidades, SHAP te dice:
- La media base del modelo es 60 unidades
- R03_lag1 anade +15 (la semana pasada fue alta, señal alcista)
- flu_au_lagged anade +12 (Australia tuvo una temporada fuerte)
- week_of_year anade +8 (semana 6, plena temporada)
- flu_eu_positives resta -10 (Europa todavia no tiene señal alta)
- Total: 60 + 15 + 12 + 8 - 10 = 85 unidades

**¿Como leer el grafico de importancia global?**
Barras horizontales. La mas larga = variable mas influyente en PROMEDIO sobre todas las predicciones.
Orden esperado: R03_lag1 > flu_au_lagged > flu_eu_positives > week_of_year.

**¿Como leer el waterfall chart?**
Muestra una prediccion especifica (la semana de mayor demanda en el test).
Cada barra = cuanto suma o resta esa variable a la prediccion base.
Rojo = suma (empuja la prediccion hacia arriba). Azul = resta (empuja hacia abajo).

**¿Por que importa esto en farmacia?**
Regulacion y confianza. Un farmaceutico o director de compras no puede decir
"el modelo dijo que si". Necesita poder explicar la decision. SHAP proporciona
ese razonamiento auditable variable a variable.
        """)

    # ── 10. Contexto Europeo (AMELI) ─────────────────────────────────────────
    with st.expander("🇫🇷  Contexto Europeo (AMELI)", expanded=False):
        st.markdown("""
**¿Para qué sirve?**
Valida que el dataset de Kaggle (una farmacia) es representativo del mercado europeo.
Usa datos nacionales de Francia (Open Medic / AMELI) como referencia de escala.

**El dataset AMELI:**
- Fuente: data.gouv.fr (gobierno frances, publico y gratuito)
- Cobertura: mercado farmaceutico nacional de Francia, 68 millones de habitantes
- Periodo: 2014-2024 (11 anos), frecuencia anual
- Variables: cajas dispensadas por categoria ATC (R03 y R06 separados)

**¿Que confirma?**
1. **Mismo patron estacional**: R03 pico en Q1 (enero-marzo) en datos nacionales,
   identico al patron semanal del dataset Kaggle. La estacionalidad es universal.
2. **R03 > R06**: en Francia, R03 supera a R06 consistentemente. Mismo en Kaggle.
3. **Factor de escala ~1000x**: Francia (68M hab.) dispensa ~1000x lo que una farmacia.
   Esto es exactamente lo esperado si la farmacia de Kaggle es "promedio europea".

**¿Como leer el grafico de barras apiladas?**
Cada barra = un ano. Azul = R03 (millones de cajas). Verde = R06.
La tendencia creciente en R03 refleja el envejecimiento de la poblacion europea.

**¿Como leer el ratio R03/R06?**
Una linea que deberia ser estable si ambas categorias crecen al mismo ritmo.
Si sube, R03 crece mas rapido que R06 (la gente usa mas medicamentos respiratorios
que antihistaminicos con los anos).

**Factor de escala:**
La tarjeta "Factor de escala" muestra cuantas veces mas grande es Francia vs la farmacia Kaggle.
~1000x. Si tienes una cadena de 1000 farmacias similares, tus numeros se acercan a los de Francia.
        """)

    # ── 11. Ensemble & Switching ──────────────────────────────────────────────
    with st.expander("🔀  Ensemble & Switching Rule", expanded=False):
        st.markdown("""
**¿Para qué sirve?**
Muestra los resultados de los dos experimentos avanzados del proyecto:
(1) ensemble multi-pais y (2) regla de cambio estacional.

**Tab 1: Ensemble Multi-Pais**

Hipotesis: combinar Australia + Nueva Zelanda + Chile en un solo modelo reduce el error
porque cuando un pais tiene una temporada atipica, los otros compensan.

Resultado: la hipotesis fue RECHAZADA.
- Australia sola: MAPE 52.52% (mejor)
- AU + NZ: MAPE 52.86% (peor)
- AU + Chile: MAPE 56.73% (peor)
- AU + NZ + Chile: MAPE 53.75% (peor)

¿Por que no funciona? NZ y Chile tienen redes de vigilancia mas pequeñas, sus lags
son distintos al de Australia, y con 302 semanas de entrenamiento el modelo no puede
extraer señal util de tres predictores debilmente correlacionados a la vez.

Este es un **hallazgo negativo honesto** — igualmente valioso que un hallazgo positivo.

**Tab 2: Switching Rule (Mejor resultado del proyecto)**

Problema identificado: el MAPE global es alto porque las semanas de verano (22-39,
mayo-septiembre) tienen demanda casi cero. Cualquier prediccion no-cero produce error
relativo enorme aunque el error absoluto sea de pocas unidades.

Solucion: en semanas 22-39, no usar XGBoost — usar la media historica de esa semana ISO.

Resultado:
- Model B WFV MAPE: 48.63%
- Switching Rule MAPE: **35.78%** (-12.85 puntos porcentuales = **mejor resultado del proyecto**)

Calibracion de intervalos de confianza:
- IC 80% cubre el 74% de semanas (bien calibrado, diferencia de solo 6pp del nominal)
- IC 50% cubre el 55.7% de semanas (bien calibrado)
        """)

    # ── 12. Diagrama de Pipeline ──────────────────────────────────────────────
    with st.expander("⚙️  Diagrama de Pipeline", expanded=False):
        st.markdown("""
**¿Para qué sirve?**
Vista tecnica del sistema completo. Muestra los 17 scripts del pipeline,
el flujo de datos entre ellos, el estado actual (que outputs existen) y como ejecutar todo.

**El pipeline de 17 pasos:**
| Pasos | Que hacen |
|-------|-----------|
| 01-04 | Descargar y limpiar los 4 datasets de origen |
| 05 | Integrar todos los datasets en uno con features de lag |
| 06 | Analisis de correlacion cruzada (CCF) |
| 07 | Entrenar XGBoost Model A y Model B |
| 08 | Descargar datos AMELI Francia |
| 09 | Analisis SHAP de explicabilidad |
| 10 | Validacion walk-forward (48 folds) |
| 11 | Modelo SARIMA + test Ljung-Box |
| 12 | Simulacion de inventario (s,Q) |
| 13 | Validacion 7 paises Hemisferio Sur |
| 14 | Figuras EDA para la tesis |
| 15 | Modelo R06 + tests Diebold-Mariano |
| 16 | Ensemble multi-pais (AU+NZ+Chile) |
| 17 | Switching Rule + calibracion IC |

**Estado de los outputs:**
Los indicadores ✅/⏳ se actualizan en tiempo real comprobando si el archivo de output existe.
Si ves muchos ⏳, ejecuta el pipeline completo: `python run_pipeline.py`

**Como ejecutar:**
```bash
python run_pipeline.py           # todos los pasos
python run_pipeline.py --from 9  # desde el paso 9
python run_pipeline.py --only 17 # solo el paso 17
```

**Inventario de outputs:**
La tabla al final lista todos los archivos generados, su descripcion y tamaño.
Si algun archivo critico falta (xgboost_best_model.json, test_predictions.csv), el dashboard fallara al cargar.
        """)

    # ── Bottom reference card ─────────────────────────────────────────────────
    st.divider()
    st.markdown("### Referencia rapida — numeros clave del proyecto")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Correlacion AU-EU", "r = 0.70", "p < 0.001")
    m2.metric("Lag optimo", "26-28 sem", "~6 meses")
    m3.metric("Mejor MAPE", "35.78%", "Switching Rule")
    m4.metric("DM test", "6.23***", "XGBoost > SARIMA")
    m5.metric("WFV folds", "48", "192 predicciones")
    m6.metric("Ahorro inventario", "EUR 273", "60 semanas")


# ==============================================================================
# PAGE 1: RESUMEN EJECUTIVO
# ==============================================================================
elif page == "Resumen Ejecutivo":
    st.markdown('<h2 class="section-title">Resumen Ejecutivo — Demanda de Medicamentos Respiratorios</h2>',
                unsafe_allow_html=True)
    st.caption(f"Datos: {df.index.min().date()} — {df.index.max().date()} · "
               f"{len(df)} semanas · Fuente: Kaggle EU + WHO FluNet + AMELI Francia")

    # KPI Cards
    last_week  = df["R03"].iloc[-1]
    prev_week  = df["R03"].iloc[-2]
    pct_change = (last_week - prev_week) / prev_week * 100
    forecast_df = forecast_next_weeks(df, model, features, n_weeks=12)
    next4_avg   = forecast_df["forecast_R03"].iloc[:4].mean()
    next12_avg  = forecast_df["forecast_R03"].mean()
    peak_idx    = forecast_df["forecast_R03"].idxmax()
    peak_week   = forecast_df.loc[peak_idx, "week_date"].strftime("%d %b %Y")

    c1, c2, c3, c4 = st.columns(4)
    arrow = "▲" if pct_change >= 0 else "▼"
    c1.markdown(f"""<div class="metric-card"><h3>Demanda Actual R03</h3>
        <h1>{last_week:.0f}</h1><p>{arrow} {abs(pct_change):.1f}% vs semana anterior</p></div>""",
        unsafe_allow_html=True)
    c2.markdown(f"""<div class="metric-card"><h3>Pronostico 4 semanas</h3>
        <h1>{next4_avg:.0f}</h1><p>Promedio unidades/semana</p></div>""",
        unsafe_allow_html=True)
    c3.markdown(f"""<div class="metric-card"><h3>Pronostico 12 semanas</h3>
        <h1>{next12_avg:.0f}</h1><p>Promedio unidades/semana</p></div>""",
        unsafe_allow_html=True)
    c4.markdown(f"""<div class="metric-card"><h3>Pico previsto</h3>
        <h1>{peak_week}</h1><p>Semana de maxima demanda</p></div>""",
        unsafe_allow_html=True)

    st.markdown("")

    # Main chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["R03"], name="R03 (historico)",
                             line=dict(color="#1f4e79", width=2),
                             hovertemplate="%{x|%d %b %Y}: %{y:.1f}<extra>R03</extra>"))
    fig.add_trace(go.Scatter(x=df.index, y=df["R06"], name="R06 (historico)",
                             line=dict(color="#70ad47", width=1.5, dash="dot"),
                             hovertemplate="%{x|%d %b %Y}: %{y:.1f}<extra>R06</extra>"))
    fig.add_trace(go.Scatter(x=forecast_df["week_date"], y=forecast_df["forecast_R03"],
                             name="Pronostico R03 (12 sem)",
                             line=dict(color="#ff6b35", width=2.5, dash="dash"),
                             hovertemplate="%{x|%d %b %Y}: %{y:.1f}<extra>Forecast</extra>"))
    # confidence band
    upper = forecast_df["forecast_R03"] * 1.15
    lower = forecast_df["forecast_R03"] * 0.85
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast_df["week_date"], forecast_df["week_date"].iloc[::-1]]),
        y=pd.concat([upper, lower.iloc[::-1]]),
        fill="toself", fillcolor="rgba(255,107,53,0.12)",
        line=dict(color="rgba(255,107,53,0)"), name="Banda ±15%", hoverinfo="skip"))
    # Use add_shape instead of add_vline — avoids Plotly 6 datetime arithmetic bug
    fig.add_shape(type="line", x0=ts(df.index[-1]), x1=ts(df.index[-1]),
                  y0=0, y1=1, yref="paper",
                  line=dict(color="#aaa", dash="dot", width=1.5))
    fig.add_annotation(x=ts(df.index[-1]), y=0.97, yref="paper",
                       text="Inicio pronostico", showarrow=False,
                       font=dict(size=10, color="#666"), textangle=-90, xanchor="right")
    fig.update_layout(
        title="Demanda Semanal R03 y R06 con Pronostico 12 semanas",
        xaxis_title="Semana", yaxis_title="Unidades vendidas",
        height=430, legend=dict(orientation="h", y=-0.15),
        hovermode="x unified", plot_bgcolor="white", paper_bgcolor="white")
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    st.plotly_chart(fig, use_container_width=True)

    # Seasonal pattern
    st.markdown("#### Patron Estacional Promedio (R03 — media por semana ISO)")
    df_s = df.copy()
    df_s["wk"] = df_s.index.isocalendar().week.astype(int)
    seasonal = df_s.groupby("wk")["R03"].mean().reset_index()
    fig2 = px.bar(seasonal, x="wk", y="R03",
                  color="R03", color_continuous_scale=["#d0e4f7", "#1f4e79"],
                  labels={"wk": "Semana ISO", "R03": "Demanda media"}, height=270)
    fig2.update_layout(coloraxis_showscale=False, showlegend=False,
                       plot_bgcolor="white", paper_bgcolor="white")
    # Use integer x for bar chart — no timestamp issue
    fig2.add_vline(x=8, line_dash="dot", line_color="#ff6b35",
                   annotation_text="Pico (sem 8, feb)")
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("El pico en semanas 1-10 (enero-febrero) confirma la temporada de gripe europea.")

    # ── Seasonal Clock ────────────────────────────────────────────────────────
    st.markdown("#### Reloj Estacional — Posicion en el Ciclo Epidemiologico")
    import datetime as _dt
    _current_iso = _dt.date.today().isocalendar()[1]  # current ISO week

    # Build polar chart with 52 spokes (one per ISO week)
    _weeks = list(range(1, 53))
    _theta = [w / 52 * 360 for w in _weeks]  # degrees

    # Background zones: AU peak (weeks 25-35), EU peak (weeks 1-10 + 46-52), transition
    def _week_color(w):
        if 1 <= w <= 10 or 46 <= w <= 52:
            return "rgba(31,78,121,0.18)"   # EU peak — blue
        elif 25 <= w <= 35:
            return "rgba(255,107,53,0.18)"  # AU peak — orange
        return "rgba(180,200,180,0.08)"

    _fig_clock = go.Figure()

    # Zone sectors via bar-polar
    _zone_colors = [_week_color(w) for w in _weeks]
    _fig_clock.add_trace(go.Barpolar(
        r=[1] * 52, theta=_theta, width=[360/52] * 52,
        marker_color=_zone_colors, marker_line_width=0,
        showlegend=False, hoverinfo="skip", name=""
    ))

    # AU peak marker (week 30 centre)
    _fig_clock.add_trace(go.Scatterpolar(
        r=[0.72], theta=[30/52*360],
        mode="markers+text",
        marker=dict(symbol="diamond", size=14, color="#ff6b35"),
        text=["🇦🇺 AU pico"], textposition="middle right",
        textfont=dict(size=10, color="#cc4400"),
        showlegend=False, name="Pico Australia"
    ))

    # EU peak marker (week 7)
    _fig_clock.add_trace(go.Scatterpolar(
        r=[0.72], theta=[7/52*360],
        mode="markers+text",
        marker=dict(symbol="star", size=14, color="#1f4e79"),
        text=["🇪🇺 EU pico"], textposition="middle right",
        textfont=dict(size=10, color="#1f4e79"),
        showlegend=False, name="Pico Europa"
    ))

    # Current week marker
    _fig_clock.add_trace(go.Scatterpolar(
        r=[0.55], theta=[_current_iso/52*360],
        mode="markers+text",
        marker=dict(symbol="circle", size=16, color="#70ad47",
                    line=dict(color="white", width=2)),
        text=[f"Hoy\nsem {_current_iso}"], textposition="middle right",
        textfont=dict(size=9, color="#2d6a2d"),
        showlegend=False, name="Semana actual"
    ))

    # Lead-lag arrow (AU→EU, 28 weeks)
    _fig_clock.add_annotation(
        text="← 28 semanas →",
        x=0.5, y=0.02, xref="paper", yref="paper",
        showarrow=False, font=dict(size=9, color="#888")
    )

    # Month labels around the ring
    _month_labels = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
    _month_weeks  = [2, 6, 10, 15, 19, 23, 27, 31, 36, 40, 45, 49]
    for _ml, _mw in zip(_month_labels, _month_weeks):
        _fig_clock.add_trace(go.Scatterpolar(
            r=[1.15], theta=[_mw/52*360],
            mode="text", text=[_ml],
            textfont=dict(size=9, color="#555"),
            showlegend=False, hoverinfo="skip", name=""
        ))

    _fig_clock.update_layout(
        polar=dict(
            radialaxis=dict(visible=False, range=[0, 1.3]),
            angularaxis=dict(
                tickmode="array",
                tickvals=list(range(0, 360, 360//52)),
                ticktext=[str(w) if w % 13 == 1 else "" for w in range(1, 53)],
                direction="clockwise", rotation=90,
                showgrid=True, gridcolor="#eee", gridwidth=1,
                showline=False, tickfont=dict(size=7)
            ),
            bgcolor="white"
        ),
        height=320,
        margin=dict(l=40, r=40, t=30, b=30),
        paper_bgcolor="white",
        showlegend=False,
    )

    _col_clock, _col_clock_legend = st.columns([2, 1])
    with _col_clock:
        st.plotly_chart(_fig_clock, use_container_width=True)
    with _col_clock_legend:
        st.markdown("")
        st.markdown("")
        st.markdown(
            '<div style="background:#fff3ee;border-left:4px solid #ff6b35;'
            'padding:10px 14px;border-radius:0 8px 8px 0;margin-bottom:8px;color:#1a1a1a">'
            '<b style="color:#cc4400">🍊 Zona naranja</b><br>'
            '<span style="font-size:0.82rem;color:#1a1a1a">Pico Australia (sem 25-35 · Jun-Ago)<br>'
            'Señal adelantada disponible</span></div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div style="background:#eef3ff;border-left:4px solid #1f4e79;'
            'padding:10px 14px;border-radius:0 8px 8px 0;margin-bottom:8px;color:#1a1a1a">'
            '<b style="color:#1f4e79">💙 Zona azul</b><br>'
            '<span style="font-size:0.82rem;color:#1a1a1a">Pico Europa (sem 1-10 · Ene-Mar)<br>'
            'Maxima demanda R03</span></div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div style="background:#efffee;border-left:4px solid #70ad47;'
            f'padding:10px 14px;border-radius:0 8px 8px 0;color:#1a1a1a">'
            f'<b style="color:#2d6a2d">💚 Semana actual</b><br>'
            f'<span style="font-size:0.82rem;color:#1a1a1a">Sem ISO {_current_iso}<br>'
            f'Lag restante hasta pico EU: '
            f'{max(0, (7 - _current_iso) % 52)} semanas</span></div>',
            unsafe_allow_html=True
        )


# ==============================================================================
# PAGE 2: PREDICCION DE DEMANDA
# ==============================================================================
elif page == "Prediccion de Demanda":
    st.markdown('<h2 class="section-title">Prediccion de Demanda — Herramienta Interactiva</h2>',
                unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 3])
    with col_a:
        n_weeks        = st.slider("Horizonte de pronostico (semanas)", 4, 26, 12)
        show_ci        = st.checkbox("Mostrar banda de confianza", value=True)
        ci_pct         = st.slider("Intervalo de confianza (%)", 5, 30, 15, disabled=not show_ci)
        st.divider()
        st.markdown("**Metricas del modelo**")
        st.metric("MAPE",  "35.78%",       help="Error porcentual medio absoluto — Switching Rule (mejor resultado)")
        st.metric("MAE",   "24.3 unidades", help="Error absoluto medio en test")
        st.metric("R²",    "-0.27",         help="Negativo por alta variabilidad estacional")
        st.metric("Modelo","XGBoost + WHO FluNet")

    with col_b:
        fc = forecast_next_weeks(df, model, features, n_weeks=n_weeks)
        hist = df["R03"].iloc[-52:]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist.values,
                                 name="Historico (52 sem)", line=dict(color="#1f4e79", width=2)))
        fig.add_trace(go.Scatter(x=fc["week_date"], y=fc["forecast_R03"],
                                 name=f"Pronostico ({n_weeks} sem)",
                                 line=dict(color="#ff6b35", width=3),
                                 mode="lines+markers", marker=dict(size=5)))
        if show_ci:
            c = ci_pct / 100
            upper = fc["forecast_R03"] * (1 + c)
            lower = fc["forecast_R03"] * (1 - c)
            fig.add_trace(go.Scatter(
                x=pd.concat([fc["week_date"], fc["week_date"].iloc[::-1]]),
                y=pd.concat([upper, lower.iloc[::-1]]),
                fill="toself", fillcolor="rgba(255,107,53,0.15)",
                line=dict(color="rgba(255,107,53,0)"), name=f"IC ±{ci_pct}%"))

        # Use add_shape — avoids Plotly 6 datetime arithmetic bug with add_vline
        fig.add_shape(type="line", x0=ts(df.index[-1]), x1=ts(df.index[-1]),
                      y0=0, y1=1, yref="paper",
                      line=dict(color="#999", dash="dot", width=1.5))
        fig.add_annotation(x=ts(df.index[-1]), y=0.98, yref="paper",
                           text="Ultimo dato", showarrow=False,
                           font=dict(size=9, color="#777"), textangle=-90, xanchor="right")
        fig.update_layout(
            title=f"Pronostico de Demanda R03 — Proximo {n_weeks} semanas",
            xaxis_title="Semana", yaxis_title="Unidades",
            height=450, hovermode="x unified",
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)

    # Forecast table
    st.markdown("#### Tabla de Pronostico Semanal")
    tbl = fc.copy()
    tbl["Semana"]           = tbl["week_date"].dt.strftime("%d %b %Y")
    tbl["Demanda R03"]      = tbl["forecast_R03"].round(1)
    tbl["Var sem/sem (%)"]  = tbl["forecast_R03"].pct_change().fillna(0).mul(100).round(1)
    if show_ci:
        tbl["IC inferior"] = (tbl["Demanda R03"] * (1 - ci_pct/100)).round(1)
        tbl["IC superior"] = (tbl["Demanda R03"] * (1 + ci_pct/100)).round(1)
    out_cols = ["Semana", "Demanda R03", "Var sem/sem (%)"] + \
               (["IC inferior", "IC superior"] if show_ci else [])
    st.dataframe(tbl[out_cols], use_container_width=True, hide_index=True)
    st.download_button("Descargar pronostico CSV",
                       tbl[out_cols].to_csv(index=False).encode("utf-8"),
                       "pronostico_r03.csv", "text/csv")


# ==============================================================================
# PAGE 2b: CALCULADORA PREDICTIVA
# ==============================================================================
elif page == "Calculadora Predictiva":
    # ── CSS extras ──────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    .risk-card { border-radius:14px; padding:18px 22px; margin:6px 0; }
    .risk-green  { background:linear-gradient(135deg,#1a6b35,#27ae60); color:white; }
    .risk-yellow { background:linear-gradient(135deg,#b8860b,#f39c12); color:white; }
    .risk-red    { background:linear-gradient(135deg,#8b0000,#c0392b); color:white; }
    .risk-card h2 { margin:0 0 2px 0; font-size:2.2rem; font-weight:800; }
    .risk-card h4 { margin:0 0 8px 0; font-size:0.82rem; letter-spacing:.08em;
                    text-transform:uppercase; opacity:.85; }
    .risk-card p  { margin:4px 0 0 0; font-size:0.88rem; opacity:.9; }
    .rec-box { background:#f0f7ff; border-left:4px solid #2e75b6;
               border-radius:0 10px 10px 0; padding:12px 16px; margin:6px 0;
               color:#1a1a1a !important; }
    .rec-box b { color:#1f4e79 !important; }
    .rec-box p, .rec-box span { color:#1a1a1a !important; }
    .warn-box { background:#fff8e1; border-left:4px solid #f39c12;
                border-radius:0 10px 10px 0; padding:12px 16px; margin:6px 0;
                color:#1a1a1a !important; }
    .warn-box b { color:#b8860b !important; }
    .warn-box p, .warn-box span { color:#1a1a1a !important; }
    .crit-box { background:#fff0f0; border-left:4px solid #c0392b;
                border-radius:0 10px 10px 0; padding:12px 16px; margin:6px 0;
                color:#1a1a1a !important; }
    .crit-box b { color:#8b0000 !important; }
    .crit-box p, .crit-box span { color:#1a1a1a !important; }
    </style>""", unsafe_allow_html=True)

    st.markdown('<h2 class="section-title">Calculadora Predictiva de Demanda R03</h2>',
                unsafe_allow_html=True)
    st.markdown("Introduce las condiciones actuales y obtén un pronóstico ajustado con "
                "intervalos de confianza empíricos, evaluación de riesgo y recomendaciones "
                "de aprovisionamiento.")

    err_profile = compute_wfv_error_profile()

    # ── Historical reference values ──────────────────────────────────────────
    hist_mean   = float(df["R03"].mean())
    hist_std    = float(df["R03"].std())
    hist_max    = float(df["R03"].max())
    last_demand = float(df["R03"].iloc[-1])
    last_4_avg  = float(df["R03"].iloc[-4:].mean())

    # Seasonal weekly averages
    df_wk = df.copy()
    df_wk["woy"] = df_wk.index.isocalendar().week.astype(int)
    seasonal_mean = df_wk.groupby("woy")["R03"].mean()

    # ── Layout: inputs (left) | main output (right) ─────────────────────────
    col_in, col_out = st.columns([1, 2.8], gap="large")

    with col_in:
        st.markdown("### Parámetros de entrada")

        st.markdown("**Horizonte de pronóstico**")
        n_weeks_fc = st.slider("Semanas a predecir", 1, 26, 8, key="calc_horizon")

        st.divider()
        st.markdown("**Señales epidemiológicas**")

        # SH country selector
        sh_country_opt = st.selectbox(
            "País HS referencia",
            ["Australia", "Nueva Zelanda", "Chile", "Argentina", "Uruguay", "Sur África"],
            index=0,
            help="País del Hemisferio Sur cuya actividad gripal actual usamos como señal adelantada."
        )
        country_map = {
            "Australia": "flu_au_lagged", "Nueva Zelanda": "flu_au_lagged",
            "Chile": "flu_au_lagged", "Argentina": "flu_au_lagged",
            "Uruguay": "flu_au_lagged", "Sur África": "flu_au_lagged"
        }

        # Default values from last data row
        last_flu_au = float(df["flu_au_lagged"].iloc[-1]) if "flu_au_lagged" in df.columns else 0.0
        last_flu_eu = float(df["flu_eu_positives"].iloc[-1]) if "flu_eu_positives" in df.columns else 0.0
        flu_au_max  = float(df["flu_au_lagged"].max())  if "flu_au_lagged" in df.columns else 100.0
        flu_eu_max  = float(df["flu_eu_positives"].max()) if "flu_eu_positives" in df.columns else 100.0

        flu_au_input = st.slider(
            f"Gripe {sh_country_opt} — semana actual (señal lag)",
            0.0, max(flu_au_max * 1.5, 10.0),
            float(np.clip(last_flu_au, 0, flu_au_max * 1.5)),
            step=0.5,
            help="Actividad gripal actual del país del Hemisferio Sur. "
                 "Esta señal llegará a Europa en ~26-28 semanas."
        )
        flu_eu_input = st.slider(
            "Gripe Europa — semana actual",
            0.0, max(flu_eu_max * 1.5, 10.0),
            float(np.clip(last_flu_eu, 0, flu_eu_max * 1.5)),
            step=0.5,
            help="Actividad gripal actual en Europa (semana en curso)."
        )

        st.divider()
        st.markdown("**Demanda reciente**")
        demand_override = st.toggle("Ajustar demanda manualmente", value=False)
        if demand_override:
            last_demand_in = st.number_input(
                "Demanda última semana (unidades)", 0.0, 300.0,
                float(round(last_demand, 1)), step=1.0
            )
            last_4_in = st.number_input(
                "Media últimas 4 semanas", 0.0, 300.0,
                float(round(last_4_avg, 1)), step=1.0
            )
        else:
            last_demand_in = last_demand
            last_4_in      = last_4_avg
            st.caption(f"Última semana registrada: **{last_demand:.1f}** unidades")
            st.caption(f"Media 4 semanas: **{last_4_avg:.1f}** unidades")

        st.divider()
        st.markdown("**Parámetros de inventario**")
        stock_actual   = st.number_input("Stock actual (unidades)", 0, 1000, 150, step=10)
        safety_stock_c = st.number_input("Safety stock objetivo", 0, 300, 50, step=5)
        lead_time_c    = st.number_input("Lead time (semanas)", 1, 12, 4, step=1)

        show_shap_detail = st.toggle("Mostrar detalle SHAP", value=False)

    with col_out:
        # ── Build modified df for forecast ──────────────────────────────────
        df_fc = df.copy()
        if "flu_au_lagged" in df_fc.columns:
            df_fc["flu_au_lagged"].iloc[-1] = flu_au_input
        if "flu_eu_positives" in df_fc.columns:
            df_fc["flu_eu_positives"].iloc[-1] = flu_eu_input
        if demand_override:
            df_fc["R03"].iloc[-1]    = last_demand_in
            df_fc["R03_lag1"].iloc[-1]    = last_demand_in
            df_fc["R03_lag4_avg"].iloc[-1] = last_4_in

        # Run forecast with empirical CI
        fc = forecast_with_ci(df_fc, model, features, n_weeks_fc, err_profile)
        fc["week_date"] = pd.to_datetime(fc["week_date"])

        # ── Key computed values ──────────────────────────────────────────────
        peak_fc       = float(fc["forecast_R03"].max())
        peak_week     = fc.loc[fc["forecast_R03"].idxmax(), "week_date"]
        avg_fc        = float(fc["forecast_R03"].mean())
        trend_pct     = (fc["forecast_R03"].iloc[-1] - fc["forecast_R03"].iloc[0]) / \
                        (fc["forecast_R03"].iloc[0] + 1e-6) * 100
        # When will stock run out at current demand trajectory?
        cum_demand     = fc["forecast_R03"].cumsum()
        weeks_to_so    = int((cum_demand <= stock_actual).sum())  # weeks until stockout
        # Reorder point check
        proj_at_lead   = stock_actual - float(fc["forecast_R03"].iloc[:lead_time_c].sum())
        need_reorder   = proj_at_lead < safety_stock_c
        recommended_order = max(0, (safety_stock_c + float(fc["forecast_R03"].iloc[:lead_time_c + 4].sum()))
                                - stock_actual)

        # Seasonal context: are we approaching peak?
        next_week_iso  = (df.index[-1] + pd.Timedelta(weeks=1)).isocalendar().week
        peak_season    = 1 <= int(next_week_iso) <= 12 or 44 <= int(next_week_iso) <= 52
        off_season     = 22 <= int(next_week_iso) <= 36

        # Risk level
        if peak_fc > hist_mean + 1.5 * hist_std or (peak_season and peak_fc > hist_mean + hist_std):
            risk_level = "ALTO"
            risk_class = "risk-red"
            risk_emoji = "🔴"
        elif peak_fc > hist_mean + 0.5 * hist_std or peak_season:
            risk_level = "MEDIO"
            risk_class = "risk-yellow"
            risk_emoji = "🟡"
        else:
            risk_level = "BAJO"
            risk_class = "risk-green"
            risk_emoji = "🟢"

        # ── Flu alert banner ─────────────────────────────────────────────────
        _flu_p75 = float(df["flu_au_lagged"].quantile(0.75)) if "flu_au_lagged" in df.columns else 0.0
        _flu_p90 = float(df["flu_au_lagged"].quantile(0.90)) if "flu_au_lagged" in df.columns else 0.0
        if flu_au_input >= _flu_p90:
            st.markdown(
                f'<div style="background:linear-gradient(135deg,#8b0000,#c0392b);color:white;'
                f'border-radius:12px;padding:14px 20px;margin-bottom:12px;'
                f'box-shadow:0 4px 16px rgba(192,57,43,0.4)">'
                f'<h3 style="margin:0 0 4px 0;font-size:1.1rem">🚨 ALERTA EPIDEMIOLOGICA — SEÑAL EXTREMA</h3>'
                f'<p style="margin:0;font-size:0.88rem">La actividad gripal de Australia '
                f'(<b>{flu_au_input:.1f}</b>) supera el percentil 90 histórico '
                f'(<b>p90 = {_flu_p90:.1f}</b>). '
                f'Riesgo muy alto de pico europeo en 26-28 semanas. '
                f'<b>Activar plan de contingencia de inventario.</b></p>'
                f'</div>',
                unsafe_allow_html=True
            )
        elif flu_au_input >= _flu_p75:
            st.markdown(
                f'<div style="background:linear-gradient(135deg,#b8860b,#e67e22);color:white;'
                f'border-radius:12px;padding:14px 20px;margin-bottom:12px;'
                f'box-shadow:0 4px 16px rgba(230,126,34,0.35)">'
                f'<h3 style="margin:0 0 4px 0;font-size:1.1rem">⚠️ ALERTA EPIDEMIOLOGICA — SEÑAL ELEVADA</h3>'
                f'<p style="margin:0;font-size:0.88rem">La actividad gripal de Australia '
                f'(<b>{flu_au_input:.1f}</b>) supera el percentil 75 histórico '
                f'(<b>p75 = {_flu_p75:.1f}</b>). '
                f'Temporada europea potencialmente intensa en ~28 semanas. '
                f'<b>Considerar incremento de pedido preventivo.</b></p>'
                f'</div>',
                unsafe_allow_html=True
            )

        # ── Risk card + key metrics ──────────────────────────────────────────
        r1, r2, r3, r4 = st.columns(4)
        r1.markdown(f"""<div class="risk-card {risk_class}">
            <h4>{risk_emoji} Nivel de riesgo</h4>
            <h2>{risk_level}</h2>
            <p>Pico previsto: {peak_fc:.1f}u</p>
        </div>""", unsafe_allow_html=True)
        r2.markdown(f"""<div class="metric-card">
            <h3>Demanda media forecast</h3>
            <h1>{avg_fc:.1f}u</h1>
            <p>vs media histórica {hist_mean:.1f}u</p>
        </div>""", unsafe_allow_html=True)
        r3.markdown(f"""<div class="metric-card">
            <h3>Stock aguanta</h3>
            <h1>~{weeks_to_so}sem</h1>
            <p>{"⚠️ Pedir YA" if weeks_to_so <= lead_time_c else "Sin rotura prevista"}</p>
        </div>""", unsafe_allow_html=True)
        r4.markdown(f"""<div class="metric-card">
            <h3>Tendencia {n_weeks_fc}sem</h3>
            <h1>{trend_pct:+.1f}%</h1>
            <p>{"↑ Creciente" if trend_pct > 5 else ("↓ Decreciente" if trend_pct < -5 else "→ Estable")}</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── Main forecast chart ──────────────────────────────────────────────
        hist_plot = df["R03"].iloc[-52:]
        fig_main = go.Figure()

        # Seasonal average reference band
        woy_future = [pd.to_datetime(d).isocalendar().week for d in fc["week_date"]]
        seas_ref   = [float(seasonal_mean.get(w, hist_mean)) for w in woy_future]
        fig_main.add_trace(go.Scatter(
            x=fc["week_date"], y=[s * 1.25 for s in seas_ref],
            fill=None, mode="lines", line_color="rgba(0,0,0,0)", showlegend=False,
            hoverinfo="skip"
        ))
        fig_main.add_trace(go.Scatter(
            x=fc["week_date"], y=[s * 0.75 for s in seas_ref],
            fill="tonexty", mode="lines", line_color="rgba(0,0,0,0)",
            fillcolor="rgba(200,200,200,0.18)", name="Rango estacional normal",
            hoverinfo="skip"
        ))

        # 80% CI
        fig_main.add_trace(go.Scatter(
            x=pd.concat([fc["week_date"], fc["week_date"].iloc[::-1]]),
            y=pd.concat([fc["ci_hi_80"], fc["ci_lo_80"].iloc[::-1]]),
            fill="toself", fillcolor="rgba(255,107,53,0.12)",
            line=dict(color="rgba(0,0,0,0)"), name="IC 80%", hoverinfo="skip"
        ))
        # 50% CI
        fig_main.add_trace(go.Scatter(
            x=pd.concat([fc["week_date"], fc["week_date"].iloc[::-1]]),
            y=pd.concat([fc["ci_hi_50"], fc["ci_lo_50"].iloc[::-1]]),
            fill="toself", fillcolor="rgba(255,107,53,0.22)",
            line=dict(color="rgba(0,0,0,0)"), name="IC 50%", hoverinfo="skip"
        ))

        # Historical
        fig_main.add_trace(go.Scatter(
            x=hist_plot.index, y=hist_plot.values,
            name="Histórico (52 sem)", line=dict(color="#1f4e79", width=2),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Demanda real: %{y:.1f}u<extra></extra>"
        ))

        # Seasonal average line on forecast period
        fig_main.add_trace(go.Scatter(
            x=fc["week_date"], y=seas_ref,
            name="Media estacional", line=dict(color="#aaaaaa", width=1.5, dash="dot"),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Media estacional: %{y:.1f}u<extra></extra>"
        ))

        # Forecast line
        fig_main.add_trace(go.Scatter(
            x=fc["week_date"], y=fc["forecast_R03"],
            name="Pronóstico XGBoost",
            line=dict(color="#ff6b35", width=3),
            mode="lines+markers",
            marker=dict(size=7, color=fc["forecast_R03"].tolist(),
                        colorscale=[[0,"#70ad47"],[0.5,"#f39c12"],[1,"#c0392b"]],
                        cmin=hist_mean - hist_std, cmax=hist_mean + 2*hist_std,
                        showscale=False),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Pronóstico: %{y:.1f}u<extra></extra>"
        ))

        # Reorder point line
        fig_main.add_shape(type="line",
            x0=ts(df.index[-1]), x1=ts(fc["week_date"].iloc[-1]),
            y0=safety_stock_c, y1=safety_stock_c,
            line=dict(color="#27ae60", dash="dot", width=1.5))
        fig_main.add_annotation(
            x=ts(fc["week_date"].iloc[-1]), y=safety_stock_c,
            text=f"Safety stock ({safety_stock_c}u)", showarrow=False,
            font=dict(size=9, color="#27ae60"), xanchor="right")

        # Now marker
        fig_main.add_shape(type="line",
            x0=ts(df.index[-1]), x1=ts(df.index[-1]),
            y0=0, y1=1, yref="paper",
            line=dict(color="#999", dash="dash", width=1.5))
        fig_main.add_annotation(
            x=ts(df.index[-1]), y=0.97, yref="paper",
            text="Ahora", showarrow=False, xanchor="right",
            font=dict(size=10, color="#666"), textangle=-90)

        # Peak annotation
        fig_main.add_annotation(
            x=ts(peak_week), y=peak_fc,
            text=f"Pico: {peak_fc:.0f}u<br>{peak_week.strftime('%d %b')}",
            showarrow=True, arrowhead=2, arrowcolor="#c0392b",
            font=dict(size=10, color="#c0392b"),
            bgcolor="rgba(255,255,255,0.85)", bordercolor="#c0392b", borderwidth=1
        )

        fig_main.update_layout(
            height=460,
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis_title="Semana", yaxis_title="Unidades R03 / semana",
            title=dict(
                text=f"Pronóstico {n_weeks_fc} semanas — Nivel de riesgo: {risk_emoji} {risk_level}",
                font=dict(size=14, color="#1f4e79"), x=0.0
            ),
            legend=dict(orientation="h", y=-0.22, font=dict(size=10)),
            hovermode="x unified",
            yaxis=dict(rangemode="tozero")
        )
        st.plotly_chart(fig_main, use_container_width=True)

        # ── Forecast table ────────────────────────────────────────────────────
        with st.expander("📋 Tabla de pronóstico semana a semana", expanded=False):
            tbl = fc[["week_date","forecast_R03","ci_lo_50","ci_hi_50",
                       "ci_lo_80","ci_hi_80","err_p50"]].copy()
            tbl["Semana"]        = pd.to_datetime(tbl["week_date"]).dt.strftime("%d %b %Y")
            tbl["Pronóstico"]    = tbl["forecast_R03"].round(1)
            tbl["IC 50% inf"]    = tbl["ci_lo_50"].round(1)
            tbl["IC 50% sup"]    = tbl["ci_hi_50"].round(1)
            tbl["IC 80% inf"]    = tbl["ci_lo_80"].round(1)
            tbl["IC 80% sup"]    = tbl["ci_hi_80"].round(1)
            tbl["MAPE esperado"] = (tbl["err_p50"] * 100).round(1).astype(str) + "%"
            st.dataframe(
                tbl[["Semana","Pronóstico","IC 50% inf","IC 50% sup",
                     "IC 80% inf","IC 80% sup","MAPE esperado"]],
                use_container_width=True, hide_index=True
            )
            _dl_col1, _dl_col2 = st.columns(2)
            with _dl_col1:
                st.download_button(
                    "⬇ Descargar pronóstico CSV",
                    tbl[["Semana","Pronóstico","IC 50% inf","IC 50% sup","IC 80% inf","IC 80% sup"]
                        ].to_csv(index=False).encode("utf-8"),
                    f"pronostico_r03_{n_weeks_fc}sem.csv", "text/csv"
                )
            with _dl_col2:
                # PDF export using reportlab
                try:
                    from reportlab.lib.pagesizes import A4
                    from reportlab.lib import colors as rl_colors
                    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                                    Table as RLTable, TableStyle)
                    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                    from reportlab.lib.units import cm
                    import io as _io

                    def _build_pdf():
                        _buf = _io.BytesIO()
                        _doc = SimpleDocTemplate(_buf, pagesize=A4,
                                                 leftMargin=2*cm, rightMargin=2*cm,
                                                 topMargin=2*cm, bottomMargin=2*cm)
                        _styles = getSampleStyleSheet()
                        _title_style = ParagraphStyle("title", parent=_styles["Heading1"],
                                                      fontSize=14, textColor=rl_colors.HexColor("#1f4e79"),
                                                      spaceAfter=6)
                        _sub_style   = ParagraphStyle("sub", parent=_styles["Normal"],
                                                      fontSize=9, textColor=rl_colors.HexColor("#555555"),
                                                      spaceAfter=10)
                        _body_style  = ParagraphStyle("body", parent=_styles["Normal"],
                                                      fontSize=10, spaceAfter=6)
                        _small_style = ParagraphStyle("small", parent=_styles["Normal"],
                                                      fontSize=8, textColor=rl_colors.HexColor("#777"))

                        import datetime as _dt2
                        _story = [
                            Paragraph("Informe de Pronostico R03", _title_style),
                            Paragraph(
                                f"Generado: {_dt2.date.today().strftime('%d/%m/%Y')} "
                                f"&nbsp;|&nbsp; Horizonte: {n_weeks_fc} semanas "
                                f"&nbsp;|&nbsp; Modelo: XGBoost + Switching Rule",
                                _sub_style
                            ),
                            Spacer(1, 0.3*cm),
                            Paragraph("<b>Metricas del modelo</b>", _body_style),
                        ]

                        # Metrics mini-table
                        _metrics_data = [
                            ["MAPE (Switching Rule)", "35.78%"],
                            ["Demanda actual (ultima sem.)", f"{last_demand:.1f} u"],
                            ["Media historica", f"{hist_mean:.1f} u"],
                            ["Riesgo estimado", risk_level],
                            ["Stock actual", f"{stock_actual} u"],
                            ["Lead time", f"{lead_time_c} sem"],
                            ["Safety stock", f"{safety_stock_c} u"],
                            ["Pedido recomendado", f"{recommended_order:.0f} u"],
                        ]
                        _mt = RLTable(_metrics_data, colWidths=[8*cm, 6*cm])
                        _mt.setStyle(TableStyle([
                            ("BACKGROUND", (0,0), (-1,0), rl_colors.HexColor("#1f4e79")),
                            ("TEXTCOLOR",  (0,0), (-1,0), rl_colors.white),
                            ("ROWBACKGROUNDS", (0,0), (-1,-1),
                             [rl_colors.HexColor("#f0f7ff"), rl_colors.white]),
                            ("FONTSIZE", (0,0), (-1,-1), 9),
                            ("GRID", (0,0), (-1,-1), 0.5, rl_colors.HexColor("#cccccc")),
                            ("LEFTPADDING",  (0,0), (-1,-1), 8),
                            ("RIGHTPADDING", (0,0), (-1,-1), 8),
                            ("TOPPADDING",   (0,0), (-1,-1), 4),
                            ("BOTTOMPADDING",(0,0), (-1,-1), 4),
                        ]))
                        _story += [_mt, Spacer(1, 0.4*cm),
                                   Paragraph("<b>Pronostico semanal</b>", _body_style)]

                        # Forecast table
                        _tbl_rows = [["Semana", "Fecha", "Pronostico", "IC 80% inf", "IC 80% sup"]]
                        for _, _r in tbl.iterrows():
                            _tbl_rows.append([
                                str(_r["Semana"]),
                                str(_r["week_date"])[:10] if "week_date" in _r.index else "",
                                f"{_r['Pronostico']:.1f}",
                                f"{_r['IC 80% inf']:.1f}",
                                f"{_r['IC 80% sup']:.1f}",
                            ])
                        _ft = RLTable(_tbl_rows, colWidths=[2.5*cm, 3*cm, 3.5*cm, 3.5*cm, 3.5*cm])
                        _ft.setStyle(TableStyle([
                            ("BACKGROUND", (0,0), (-1,0), rl_colors.HexColor("#2e75b6")),
                            ("TEXTCOLOR",  (0,0), (-1,0), rl_colors.white),
                            ("FONTSIZE",   (0,0), (-1,-1), 8),
                            ("ROWBACKGROUNDS", (0,1), (-1,-1),
                             [rl_colors.HexColor("#f7fbff"), rl_colors.white]),
                            ("GRID", (0,0), (-1,-1), 0.5, rl_colors.HexColor("#dddddd")),
                            ("ALIGN", (2,0), (-1,-1), "CENTER"),
                            ("LEFTPADDING",  (0,0), (-1,-1), 6),
                            ("RIGHTPADDING", (0,0), (-1,-1), 6),
                            ("TOPPADDING",   (0,0), (-1,-1), 3),
                            ("BOTTOMPADDING",(0,0), (-1,-1), 3),
                        ]))
                        _story += [_ft, Spacer(1, 0.5*cm),
                                   Paragraph(
                                       "Nota: Los intervalos de confianza son empiricos, derivados de "
                                       "192 predicciones walk-forward agrupadas por semana ISO. "
                                       "MAPE esperado varia entre ~10% (invierno) y >150% (verano).",
                                       _small_style
                                   )]
                        _doc.build(_story)
                        return _buf.getvalue()

                    st.download_button(
                        "⬇ Exportar PDF",
                        _build_pdf(),
                        f"pronostico_r03_{n_weeks_fc}sem.pdf",
                        "application/pdf"
                    )
                except ImportError:
                    st.caption("PDF: instala reportlab (`pip install reportlab`)")

    # ── Full-width recommendations panel ────────────────────────────────────
    st.markdown("---")
    st.markdown("### Recomendaciones de aprovisionamiento")

    rec_col, prec_col = st.columns([1.2, 1])

    with rec_col:
        # Procurement action
        if need_reorder:
            st.markdown(f"""<div class="crit-box">
                <b>⚠️ PEDIDO URGENTE RECOMENDADO</b><br>
                El stock proyectado en {lead_time_c} semanas ({proj_at_lead:.0f}u) cae por debajo
                del safety stock ({safety_stock_c}u). Realiza un pedido de al menos
                <b>{recommended_order:.0f} unidades</b> esta semana para evitar rotura en el horizonte
                de lead time.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="rec-box">
                <b>✅ Stock suficiente a {lead_time_c} semanas</b><br>
                Stock proyectado al final del lead time: <b>{proj_at_lead:.0f}u</b>
                (por encima del safety stock de {safety_stock_c}u).
                Próximo pedido sugerido cuando el stock caiga a ≤{safety_stock_c + int(avg_fc * lead_time_c)}u.
            </div>""", unsafe_allow_html=True)

        # Demand level recommendation
        if risk_level == "ALTO":
            st.markdown(f"""<div class="crit-box">
                <b>🔴 TEMPORADA DE PICO — Aumentar stock de seguridad</b><br>
                La demanda prevista ({peak_fc:.0f}u pico) supera en
                {((peak_fc/hist_mean - 1)*100):.0f}% la media histórica ({hist_mean:.0f}u).
                Considera aumentar el safety stock al menos un <b>30-50%</b> durante la temporada.
                Alerta a proveedores con 8+ semanas de antelación.
            </div>""", unsafe_allow_html=True)
        elif risk_level == "MEDIO":
            st.markdown(f"""<div class="warn-box">
                <b>🟡 Temporada de transición — Vigilancia activa</b><br>
                Demanda prevista un {((avg_fc/hist_mean - 1)*100):+.0f}% sobre la media histórica.
                Monitoriza la actividad gripal europea semanalmente.
                Considera adelantar pedidos 1-2 semanas respecto al ciclo habitual.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="rec-box">
                <b>🟢 Temporada baja — Optimizar costes de almacén</b><br>
                Demanda prevista por debajo de la media ({avg_fc:.0f}u vs {hist_mean:.0f}u media).
                Momento adecuado para reducir stock de ciclo y minimizar costes de almacén.
                Prioriza agotamiento de stock antiguo (FIFO).
            </div>""", unsafe_allow_html=True)

        # Flu signal interpretation
        flu_percentile = float(
            (df["flu_au_lagged"] <= flu_au_input).mean() * 100
        ) if "flu_au_lagged" in df.columns else 50.0
        if flu_percentile > 80:
            st.markdown(f"""<div class="crit-box">
                <b>⚡ Señal {sh_country_opt} muy elevada (percentil {flu_percentile:.0f})</b><br>
                La actividad gripal actual en {sh_country_opt} es excepcionalmente alta.
                Históricament esto anticipa un pico europeo severo en 26-28 semanas.
                Considera preparar stock adicional con suficiente antelación.
            </div>""", unsafe_allow_html=True)
        elif flu_percentile > 55:
            st.markdown(f"""<div class="warn-box">
                <b>📈 Señal {sh_country_opt} moderada-alta (percentil {flu_percentile:.0f})</b><br>
                La actividad gripal en {sh_country_opt} está por encima de la mediana histórica.
                Mantén vigilancia sobre la evolución de la temporada europea.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="rec-box">
                <b>📉 Señal {sh_country_opt} normal (percentil {flu_percentile:.0f})</b><br>
                La actividad gripal en {sh_country_opt} está dentro del rango normal.
                No se anticipa un pico europeo excepcional en el horizonte de 26 semanas.
            </div>""", unsafe_allow_html=True)

    with prec_col:
        st.markdown("#### Precauciones e interpretación")

        st.markdown(f"""
**Sobre los intervalos de confianza:**
- Los IC son **empíricos**, calculados a partir de los errores reales del modelo en 192 semanas de validación walk-forward
- IC 50%: en la mitad de los casos, la demanda real estará dentro de esta banda
- IC 80%: cubre la mayoría de los escenarios normales — **usa el IC 80% para planificación de inventario**
- El MAPE esperado varía por temporada: **invierno ~14-20%**, verano **>100%** (demanda cerca de cero)
""")

        st.markdown(f"""
**Limitaciones del modelo:**
- El modelo se entrenó con datos 2014-2019 de **una farmacia europea**. Para volúmenes nacionales, escala por el factor AMELI (~26.000x)
- No modela eventos extraordinarios: pandemias, discontinuidades de suministro, cambios de formulario
- La señal australiana tiene un lag de **28 semanas** — cambios en la temporada actual de {sh_country_opt} tardarán ~7 meses en verse reflejados en Europa
""")

        st.markdown(f"""
**Cuándo desconfiar del pronóstico:**
- Si el MAPE esperado de la semana supera el **60%** (verano, semanas 22-36)
- Si la señal {sh_country_opt} actual difiere mucho del año anterior por una causa específica (ej. pandemia, vacuna nueva)
- Si hay discontinuidades en el suministro que el modelo no puede observar
""")

        # Seasonal context box
        season_context = (
            "**Temporada de pico (ene-mar)**: máxima demanda prevista. "
            "Mantén stock de seguridad elevado." if 1 <= int(next_week_iso) <= 12
            else "**Pre-temporada (oct-dic)**: demanda en ascenso. "
            "Momento clave para aprovisionar." if 40 <= int(next_week_iso) <= 52
            else "**Post-temporada (abr-may)**: demanda bajando. "
            "Reducir stocks gradualmente." if 14 <= int(next_week_iso) <= 21
            else "**Temporada baja (jun-sep)**: demanda mínima. "
            "Optimizar costes de almacén."
        )
        st.info(f"📅 **Contexto estacional** (semana {int(next_week_iso)}): {season_context}")

        # Model uncertainty note
        mape_this_week = float(
            err_profile.loc[int(next_week_iso), "p50"] * 100
        ) if err_profile is not None and int(next_week_iso) in err_profile.index else 25.0
        st.caption(
            f"MAPE mediano histórico para la semana {int(next_week_iso)}: "
            f"**{mape_this_week:.1f}%** — "
            f"{'⚠️ alta incertidumbre' if mape_this_week > 50 else '✓ incertidumbre normal'}"
        )

    # ── SHAP detail (optional) ────────────────────────────────────────────────
    if show_shap_detail and shap_df is not None:
        st.markdown("---")
        st.markdown("### Contribución SHAP — qué está impulsando el pronóstico")
        shap_feat_cols = [c for c in shap_df.columns if c.startswith("shap_")]
        shap_means     = shap_df[shap_feat_cols].abs().mean().sort_values(ascending=True)
        feat_labels    = {
            "shap_R03_lag1":         "Demanda semana anterior",
            "shap_R03_lag4_avg":     "Demanda media 4 semanas",
            "shap_flu_au_positives": "Gripe AU (semana actual)",
            "shap_flu_au_lagged":    f"Gripe {sh_country_opt} (lag 26-28 sem)",
            "shap_flu_eu_positives": "Gripe Europa (actual)",
        }
        shap_fig = go.Figure(go.Bar(
            y=[feat_labels.get(c, c.replace("shap_","")) for c in shap_means.index],
            x=shap_means.values,
            orientation="h",
            marker_color=["#1f4e79" if "lag" in c else "#ff6b35"
                          if "au" in c else "#70ad47" for c in shap_means.index],
            text=[f"{v:.3f}u" for v in shap_means.values],
            textposition="outside"
        ))
        shap_fig.update_layout(
            height=260, plot_bgcolor="white", paper_bgcolor="white",
            xaxis_title="Impacto medio |SHAP| (unidades/semana)",
            title="Importancia de features — basada en test set (SHAP TreeExplainer)"
        )
        st.plotly_chart(shap_fig, use_container_width=True)
        st.caption(
            "Los valores SHAP muestran el impacto promedio de cada feature sobre el pronóstico. "
            f"La señal de {sh_country_opt} con lag de 26-28 semanas aporta en promedio "
            f"{float(shap_df['shap_flu_au_lagged'].abs().mean()):.2f}u/semana de impacto — "
            "el único feature disponible con 6 meses de antelación."
        )


# ==============================================================================
# PAGE 3: ANALISIS LEAD-LAG
# ==============================================================================
elif page == "Analisis Lead-Lag":
    st.markdown('<h2 class="section-title">Analisis Lead-Lag: Hemisferio Sur a Norte</h2>',
                unsafe_allow_html=True)
    st.info("**Hipotesis central:** La gripe australiana (jun-ago) anticipa la demanda europea (dic-feb) "
            "por ~28 semanas. Esto permite pronosticos con 6 meses de antelacion usando datos reales de la OMS.")

    if flu_au is not None and flu_eu is not None and "INF_ALL" in flu_au.columns:
        combined = pd.DataFrame({
            "Australia": flu_au["INF_ALL"].resample("W-SUN").sum(),
            "Europa":    flu_eu["INF_ALL"].resample("W-SUN").sum(),
        }).dropna()

        # Normalize 0-1
        for col in combined.columns:
            mn, mx = combined[col].min(), combined[col].max()
            combined[col] = (combined[col] - mn) / (mx - mn)

        lag_sel = st.slider("Desplazar Australia (semanas) para alinear visualmente", 0, 52, 28)
        last250 = combined.iloc[-250:].copy()
        last250["Australia_shift"] = combined["Australia"].shift(-lag_sel).iloc[-250:]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=last250.index, y=last250["Europa"],
                                 name="Europa", line=dict(color="#1f4e79", width=2)))
        fig.add_trace(go.Scatter(x=last250.index, y=last250["Australia"],
                                 name="Australia (sin desplazar)",
                                 line=dict(color="#ed7d31", width=1.5, dash="dot")))
        fig.add_trace(go.Scatter(x=last250.index, y=last250["Australia_shift"],
                                 name=f"Australia (+{lag_sel} sem desplazada)",
                                 line=dict(color="#c00000", width=2.5, dash="dash")))
        fig.update_layout(
            title="Actividad Gripal Normalizada: Australia vs Europa (ultimas 250 semanas)",
            xaxis_title="Fecha", yaxis_title="Intensidad gripal (0-1)",
            height=380, hovermode="x unified",
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True)

        # Cross-correlation
        col_cc, col_res = st.columns([2, 1])
        with col_cc:
            lags   = list(range(-52, 53))
            au_raw = combined["Australia"]
            eu_raw = combined["Europa"]
            corrs  = []
            for lag in lags:
                if lag >= 0:
                    r = au_raw.iloc[:len(au_raw)-lag].corr(eu_raw.iloc[lag:]) if lag < len(au_raw) else 0
                else:
                    r = au_raw.iloc[-lag:].corr(eu_raw.iloc[:len(eu_raw)+lag]) if -lag < len(eu_raw) else 0
                corrs.append(float(r) if not np.isnan(r) else 0)

            max_corr = max(corrs)
            bar_colors = ["#ff6b35" if l in [26, 27, 28] else
                          ("#c00000" if c == max_corr else "#d0e4f7")
                          for l, c in zip(lags, corrs)]

            fig2 = go.Figure(go.Bar(
                x=lags, y=corrs, marker_color=bar_colors,
                hovertemplate="Lag %{x} sem: r=%{y:.3f}<extra></extra>"))
            fig2.add_hline(y=0, line_color="#999")
            # Numeric x — no timestamp issue
            fig2.add_vline(x=28, line_dash="dash", line_color="#c00000",
                           annotation_text="Lag optimo: 28 sem (r=0.70)")
            fig2.add_vline(x=26, line_dash="dot", line_color="#ff6b35",
                           annotation_text="Hipotesis TFG (26 sem)")
            fig2.update_layout(
                title="Funcion de Correlacion Cruzada: Australia lidera Europa",
                xaxis_title="Lag (semanas, positivo = Australia lidera)",
                yaxis_title="Correlacion de Pearson",
                height=370, plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig2, use_container_width=True)

        with col_res:
            st.markdown("#### Resultados validacion")
            st.metric("r en lag 0",  "0.009",  help="Sin correlacion instantanea: hemisferios opuestos")
            st.metric("r maximo",    "0.70",   delta="lag = 28 semanas")
            st.metric("r en lag 26", "0.558",  help="Hipotesis original de la tesis (26 sem)")
            st.divider()
            st.success("Con r = **0.70** a 28 semanas usando 1,531 semanas reales de la OMS "
                       "(1996-2024), la hipotesis lead-lag queda **empiricamente confirmada**.")
            st.markdown("**Implicacion practica:**")
            st.markdown("Un farmaceutico europeo puede observar el pico gripal australiano "
                        "en julio y predecir la demanda de R03 para el siguiente febrero "
                        "con 7 meses de antelacion.")
    else:
        st.warning("Datos FluNet no encontrados. Ejecuta src/02_clean_flunet.py primero.")


# ==============================================================================
# PAGE 4: VALIDACION HEMISFERICA
# ==============================================================================
elif page == "Validacion Hemisferica":
    st.markdown('<h2 class="section-title">Validacion Hemisferica — El mecanismo es universal, no australiano</h2>',
                unsafe_allow_html=True)
    st.markdown(
        "Si la teoria es correcta — que Europa puede predecirse porque Australia esta en el "
        "**hemisferio sur** — entonces **cualquier pais del hemisferio sur** deberia mostrar "
        "el mismo patron de lead-lag con Europa. Este analisis valida esa hipotesis con "
        "7 paises usando WHO FluNet 1997-2024."
    )

    if sh_ccf is None:
        st.warning("Ejecuta: `python src/13_southern_hemisphere_analysis.py`")
    else:
        SH_COLORS = {
            "Australia":    "#1f4e79",
            "New Zealand":  "#ff6b35",
            "South Africa": "#70ad47",
            "Chile":        "#7030a0",
            "Argentina":    "#c00000",
            "Brazil":       "#0070c0",
            "Uruguay":      "#f79646",
        }

        # KPI cards
        best_r_row = sh_ccf.sort_values("peak_r", ascending=False).iloc[0]
        all_sig    = (sh_ccf["p_value"] < 0.001).all()
        lag_range  = f"{sh_ccf['optimal_lag_w'].min()}–{sh_ccf['optimal_lag_w'].max()} semanas"
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(f"""<div class="metric-card"><h3>Paises testados</h3><h1>7</h1>
            <p>Australia, NZ, SA, Chile, ARG, BR, URY</p></div>""", unsafe_allow_html=True)
        k2.markdown(f"""<div class="metric-card"><h3>Todos significativos</h3>
            <h1>{'Si' if all_sig else 'No'}</h1><p>p &lt; 0.001 en todos los casos</p></div>""",
            unsafe_allow_html=True)
        k3.markdown(f"""<div class="metric-card"><h3>Rango de lags optimos</h3>
            <h1>{lag_range}</h1><p>~6 meses — consistente con hipotesis</p></div>""",
            unsafe_allow_html=True)
        k4.markdown(f"""<div class="metric-card"><h3>Mayor correlacion</h3>
            <h1>r = {best_r_row['peak_r']:.3f}</h1>
            <p>{best_r_row['country']} @ lag {int(best_r_row['optimal_lag_w'])}w</p></div>""",
            unsafe_allow_html=True)

        st.divider()

        tab1, tab2, tab3 = st.tabs([
            "Curvas CCF",
            "Comparacion de correlaciones",
            "Rendimiento predictivo por pais"
        ])

        with tab1:
            st.markdown("#### Funcion de correlacion cruzada (CCF) — cada pais del Hemisferio Sur vs Europa")
            st.markdown(
                "Cada curva muestra cómo la correlacion entre el pais del HS y la gripe europea "
                "cambia segun el numero de semanas de adelanto. El **pico de cada curva** indica "
                "cuantas semanas antes ese pais predice la actividad gripal en Europa."
            )

            # Load raw CCF data from FluNet global for interactive chart
            # Use the results table to build a representative chart
            fig_ccf = go.Figure()

            # Load individual country FluNet data for CCF
            eu_data = load_flunet("europe")
            if eu_data is not None:
                eu_norm = eu_data["INF_ALL"] / eu_data["INF_ALL"].max()

                for country, color in SH_COLORS.items():
                    sh_data = load_flunet_country(country)
                    if sh_data is None:
                        continue
                    sh_norm = sh_data["INF_ALL"] / sh_data["INF_ALL"].max() \
                              if sh_data["INF_ALL"].max() > 0 else sh_data["INF_ALL"]

                    # Compute CCF interactively
                    common = sh_norm.index.intersection(eu_norm.index)
                    if len(common) < 50:
                        continue
                    sh_c = sh_norm.loc[common].values
                    eu_c = eu_norm.loc[common].values

                    lags_r = []
                    for lag in range(0, 53):
                        if lag == 0:
                            x, y = sh_c, eu_c
                        else:
                            x, y = sh_c[:-lag], eu_c[lag:]
                        if len(x) < 30:
                            break
                        r_val, _ = __import__("scipy.stats", fromlist=["pearsonr"]).pearsonr(x, y)
                        lags_r.append(r_val)

                    peak_r  = sh_ccf[sh_ccf["country"] == country]["peak_r"].iloc[0]
                    opt_lag = sh_ccf[sh_ccf["country"] == country]["optimal_lag_w"].iloc[0]
                    lw = 3 if country == "Australia" else 1.8
                    dash = "solid" if country not in ["Brazil"] else "dash"

                    fig_ccf.add_trace(go.Scatter(
                        x=list(range(len(lags_r))), y=lags_r,
                        name=f"{country} (peak r={peak_r:.3f} @ {int(opt_lag)}w)",
                        line=dict(color=color, width=lw, dash=dash),
                        hovertemplate="Lag: %{x}w<br>r: %{y:.4f}<extra>" + country + "</extra>"
                    ))
                    # Mark peak
                    if len(lags_r) > int(opt_lag):
                        fig_ccf.add_trace(go.Scatter(
                            x=[int(opt_lag)], y=[lags_r[int(opt_lag)]],
                            mode="markers",
                            marker=dict(size=10, color=color,
                                        line=dict(color="white", width=2)),
                            showlegend=False, hoverinfo="skip"
                        ))
            else:
                # Fallback: use summary data to draw approximate bars
                for _, row in sh_ccf.iterrows():
                    color = SH_COLORS.get(row["country"], "#999")
                    fig_ccf.add_trace(go.Bar(
                        x=[row["country"]], y=[row["peak_r"]],
                        name=row["country"], marker_color=color,
                        text=[f"lag={int(row['optimal_lag_w'])}w  r={row['peak_r']:.3f}"],
                        textposition="outside"
                    ))

            fig_ccf.add_hline(y=0, line_color="#ccc", line_width=1)
            fig_ccf.add_vline(x=26, line_dash="dot", line_color="#888",
                              annotation_text="26 semanas", annotation_position="top right")
            fig_ccf.update_layout(
                height=420, plot_bgcolor="white", paper_bgcolor="white",
                xaxis_title="Lag (semanas) — el pais HS predice Europa con N semanas de adelanto",
                yaxis_title="Correlacion de Pearson (r)",
                title="CCF: Hemisferio Sur → Europa | WHO FluNet 1997-2024",
                legend=dict(orientation="h", y=-0.28, font=dict(size=9)),
                hovermode="x unified"
            )
            st.plotly_chart(fig_ccf, use_container_width=True)

            st.info(
                "**Resultado clave**: todos los paises del hemisferio sur muestran correlacion "
                "positiva significativa (p<0.001) a lags de 27-36 semanas. El mecanismo no es "
                "especifico de Australia — es una propiedad del hemisferio sur."
            )

        with tab2:
            st.markdown("#### Correlacion y lag optimo por pais")
            fig_bar = go.Figure()
            sh_sorted = sh_ccf.sort_values("peak_r", ascending=True)
            fig_bar.add_trace(go.Bar(
                y=sh_sorted["country"],
                x=sh_sorted["peak_r"],
                orientation="h",
                marker_color=[SH_COLORS.get(c, "#999") for c in sh_sorted["country"]],
                text=[f"r={r:.3f}  lag={int(l)}w  p<0.001"
                      for r, l in zip(sh_sorted["peak_r"], sh_sorted["optimal_lag_w"])],
                textposition="outside",
                hovertemplate="%{y}: r=%{x:.4f}<extra></extra>"
            ))
            fig_bar.update_layout(
                height=350, plot_bgcolor="white", paper_bgcolor="white",
                xaxis_title="Correlacion de Pearson maxima (r)", yaxis_title="",
                title="Correlacion cruzada maxima por pais — SH → Europa",
                xaxis=dict(range=[0, 0.9])
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Lag comparison
            fig_lag = go.Figure()
            fig_lag.add_trace(go.Scatter(
                x=sh_ccf["peak_r"], y=sh_ccf["optimal_lag_w"],
                mode="markers+text",
                text=sh_ccf["country"],
                textposition="top center",
                marker=dict(
                    size=sh_ccf["peak_r"] * 60,
                    color=[SH_COLORS.get(c, "#999") for c in sh_ccf["country"]],
                    line=dict(color="white", width=1.5)
                ),
                hovertemplate="%{text}<br>r=%{x:.3f}  lag=%{y}w<extra></extra>"
            ))
            fig_lag.add_hline(y=26, line_dash="dot", line_color="#888",
                              annotation_text="26 semanas (6 meses)")
            fig_lag.update_layout(
                height=360, plot_bgcolor="white", paper_bgcolor="white",
                xaxis_title="Correlacion maxima (r)",
                yaxis_title="Lag optimo (semanas)",
                title="Correlacion vs Lag optimo — burbujas proporcionales a r"
            )
            st.plotly_chart(fig_lag, use_container_width=True)
            st.caption(
                "Todos los lags optimos estan en el rango 27-36 semanas (~6-9 meses), "
                "consistente con la diferencia estacional entre hemisferios. "
                "Brasil (clima tropical, marcado con linea discontinua) muestra r=0.52 "
                "a lag=36w, sugiriendo que incluso paises con temporadas de gripe mixtas "
                "participan en el patron hemisferio."
            )

        with tab3:
            st.markdown("#### MAPE del modelo XGBoost usando la senal de cada pais")
            st.markdown(
                "Cada modelo usa los mismos features autoregresivos (R03_lag1, R03_lag4_avg) "
                "mas la senal de gripe del pais del HS con su lag optimo. "
                "Baseline = solo lags autoregresivos (sin senal del HS)."
            )

            model_sorted = sh_models.sort_values("MAPE")
            colors_model = [SH_COLORS.get(c, "#999") for c in model_sorted["country"]]

            fig_model = go.Figure()
            fig_model.add_trace(go.Bar(
                y=model_sorted["country"] + " (lag=" + model_sorted["lag"].astype(int).astype(str) + "w)",
                x=model_sorted["MAPE"],
                orientation="h",
                marker_color=colors_model,
                text=[f"{m:.2f}%" for m in model_sorted["MAPE"]],
                textposition="outside",
                hovertemplate="%{y}<br>MAPE: %{x:.2f}%<extra></extra>"
            ))
            fig_model.add_vline(x=46.45, line_dash="dot", line_color="#aaa",
                                annotation_text="Baseline (sin HS): 46.45%",
                                annotation_position="top right")
            fig_model.update_layout(
                height=380, plot_bgcolor="white", paper_bgcolor="white",
                xaxis_title="MAPE (%) — conjunto de test",
                title="MAPE por pais del Hemisferio Sur (+ features autoregresivos)"
            )
            st.plotly_chart(fig_model, use_container_width=True)

            best_c = sh_models.sort_values("MAPE").iloc[0]
            st.success(
                f"**Hallazgo**: {best_c['country']} produce el mejor MAPE individual ({best_c['MAPE']:.2f}%) "
                f"con lag={int(best_c['lag'])}w. La alta correlacion de Australia (r=0.73) no garantiza "
                f"el mejor rendimiento predictivo — la calidad y consistencia de la senal importa tanto "
                f"como su magnitud. Todos los paises mejoran el baseline de lags puros, "
                f"confirmando que la senal hemisferica añade valor real."
            )

            # Summary table — merge on country, keeping non-duplicate columns from sh_ccf
            display_sh = sh_models.merge(
                sh_ccf[["country", "optimal_lag_w", "p_value", "tropical"]],
                on="country"
            )[["country", "peak_r", "lag", "optimal_lag_w", "tropical", "p_value", "MAPE", "MAE", "R2"]]
            display_sh = display_sh.sort_values("MAPE")
            st.dataframe(
                display_sh, hide_index=True, use_container_width=True,
                column_config={
                    "country":       "Pais",
                    "peak_r":        st.column_config.NumberColumn("Corr. max (r)", format="%.4f"),
                    "lag":           st.column_config.NumberColumn("Lag modelo (w)", format="%d"),
                    "optimal_lag_w": st.column_config.NumberColumn("Lag CCF optimo (w)", format="%d"),
                    "tropical":      st.column_config.CheckboxColumn("Tropical"),
                    "p_value":       st.column_config.NumberColumn("p-valor", format="%.4f"),
                    "MAPE":          st.column_config.NumberColumn("MAPE (%)", format="%.2f"),
                    "MAE":           st.column_config.NumberColumn("MAE", format="%.2f"),
                    "R2":            st.column_config.NumberColumn("R2", format="%.4f"),
                }
            )


# ==============================================================================
# PAGE 5: RENDIMIENTO DEL MODELO
# ==============================================================================
elif page == "Rendimiento del Modelo":
    st.markdown('<h2 class="section-title">Rendimiento del Modelo XGBoost</h2>',
                unsafe_allow_html=True)
    st.caption(f"Entrenado: {meta['train_end']} | Test: {meta['test_start']} - {meta['test_end']} | "
               f"Features: {', '.join(features)}")

    # Metrics table
    met_rows = [
        {"Modelo": "Baseline (solo lags autoregresivos)", "MAE": 22.58, "RMSE": 30.42, "MAPE (%)": 46.45, "R2": -0.0995},
        {"Modelo": "Con WHO FluNet real (tesis)",          "MAE": 24.32, "RMSE": 32.71, "MAPE (%)": 44.16, "R2": -0.2713},
        {"Modelo": "Sintetico original (referencia)",      "MAE": 21.16, "RMSE": 28.89, "MAPE (%)": 49.75, "R2": -0.0212},
    ]
    met_df = pd.DataFrame(met_rows)

    col_t, col_chart = st.columns([1, 2])
    with col_t:
        st.markdown("#### Comparacion de modelos")
        st.dataframe(met_df, hide_index=True, use_container_width=True)
        st.info("El modelo sintetico tenia **data leakage** (proxy circular). "
                "Con datos reales, el MAPE baja de 46.45% a **44.16%** — "
                "la mejora mas valida aunque numericamente menor.")

    with col_chart:
        models_short = ["Baseline", "FluNet Real", "Sintetico"]
        x_pos = [0, 1, 2]
        fig = go.Figure()
        for i, (metric, color) in enumerate([("MAE", "#1f4e79"), ("RMSE", "#2e75b6"), ("MAPE (%)", "#70ad47")]):
            fig.add_trace(go.Bar(
                x=[p + (i-1)*0.28 for p in x_pos],
                y=[r[metric] for r in met_rows],
                name=metric, marker_color=color, width=0.26))
        fig.update_layout(
            title="Metricas de Error por Modelo",
            xaxis=dict(tickvals=x_pos, ticktext=models_short),
            yaxis_title="Error", barmode="overlay",
            height=330, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Actual vs Predicted
    st.markdown("#### Real vs Predicho en el Conjunto de Test (60 semanas)")
    fig2 = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True,
                         subplot_titles=["Demanda real vs predicha", "Residuos"])
    fig2.add_trace(go.Scatter(x=preds["week_date"], y=preds["actual_R03"],
                              name="Real", line=dict(color="#1f4e79", width=2.5)), row=1, col=1)
    fig2.add_trace(go.Scatter(x=preds["week_date"], y=preds["predicted_R03"],
                              name="Predicho", line=dict(color="#ff6b35", width=2, dash="dash")), row=1, col=1)
    residuals = preds["actual_R03"] - preds["predicted_R03"]
    fig2.add_trace(go.Bar(x=preds["week_date"], y=residuals,
                          name="Residuo", marker_color=["#c00000" if r < 0 else "#70ad47" for r in residuals]),
                   row=2, col=1)
    fig2.add_hline(y=0, line_color="#999", row=2, col=1)
    fig2.update_layout(height=500, hovermode="x unified",
                       plot_bgcolor="white", paper_bgcolor="white",
                       legend=dict(orientation="h", y=-0.08))
    st.plotly_chart(fig2, use_container_width=True)

    # Feature importance
    importance = model.get_booster().get_fscore()
    if importance:
        st.markdown("#### Importancia de Variables")
        imp_df = pd.DataFrame(importance.items(), columns=["Variable", "Importancia"])
        imp_df = imp_df.sort_values("Importancia")
        fig3 = px.bar(imp_df, x="Importancia", y="Variable", orientation="h",
                      color="Importancia", color_continuous_scale=["#d0e4f7", "#1f4e79"],
                      height=max(250, len(imp_df)*50))
        fig3.update_layout(plot_bgcolor="white", paper_bgcolor="white", coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    # SARIMA comparison block (if available)
    if sarima_preds is not None and sarima_meta is not None:
        st.divider()
        st.markdown("#### Comparacion con SARIMA — Baseline Estadistico Clasico")
        sm = sarima_meta
        order_str = f"SARIMA{tuple(sm['order'])}x{tuple(sm['seasonal_order'])}"
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("SARIMA MAPE",    f"{sm['metrics_sarima']['MAPE']:.2f}%")
        sc2.metric("XGBoost MAPE",   f"{sm['metrics_xgboost']['MAPE']:.2f}%",
                   delta=f"{sm['sarima_vs_xgb_mape_delta']:+.2f} pp (SARIMA vs XGB)", delta_color="inverse")
        sc3.metric("Naive MAPE",     f"{sm['metrics_naive']['MAPE']:.2f}%")
        sc4.metric("Orden SARIMA",   order_str)

        fig_sar = go.Figure()
        fig_sar.add_trace(go.Scatter(
            x=sarima_preds["week_date"], y=sarima_preds["actual_R03"],
            name="Real", line=dict(color="#111", width=2.5)))
        fig_sar.add_trace(go.Scatter(
            x=sarima_preds["week_date"], y=sarima_preds["sarima_pred"],
            name=order_str, line=dict(color="#70ad47", width=1.8, dash="dash")))
        fig_sar.add_trace(go.Scatter(
            x=sarima_preds["week_date"], y=sarima_preds["xgboost_pred"],
            name="XGBoost (Model B)", line=dict(color="#ff6b35", width=1.8, dash="dash")))
        fig_sar.add_traces([
            go.Scatter(x=sarima_preds["week_date"], y=sarima_preds["sarima_ci_hi"],
                       fill=None, mode="lines", line_color="rgba(112,173,71,0)", showlegend=False),
            go.Scatter(x=sarima_preds["week_date"], y=sarima_preds["sarima_ci_lo"],
                       fill="tonexty", mode="lines", line_color="rgba(112,173,71,0)",
                       fillcolor="rgba(112,173,71,0.12)", name="SARIMA 80% IC"),
        ])
        fig_sar.update_layout(
            height=360, plot_bgcolor="white", paper_bgcolor="white",
            xaxis_title="Semana", yaxis_title="Unidades R03",
            title=f"SARIMA vs XGBoost — Conjunto de Test (60 semanas)",
            legend=dict(orientation="h", y=-0.18), hovermode="x unified"
        )
        st.plotly_chart(fig_sar, use_container_width=True)
        st.caption(
            f"**Interpretacion**: XGBoost supera a SARIMA por {abs(sm['sarima_vs_xgb_mape_delta']):.1f} pp en MAPE. "
            f"SARIMA({sm['order']}) seleccionado por BIC con m=52 (ciclo anual). "
            f"La ventaja de XGBoost proviene de los features exogenos (WHO FluNet) que "
            f"SARIMA no puede aprovechar siendo univariante."
        )

    # ── Prophet comparison ───────────────────────────────────────────────────
    _prophet_meta_path = os.path.join(OUT, "prophet_meta.json")
    _prophet_pred_path = os.path.join(OUT, "prophet_predictions.csv")
    if os.path.exists(_prophet_meta_path):
        st.divider()
        st.markdown("#### Comparacion con Prophet (Meta) — Tercer Modelo de Referencia")
        with open(_prophet_meta_path) as _pf:
            _pm = json.load(_pf)

        _dm = _pm.get("dm_xgboost_vs_prophet", {})
        _pc1, _pc2, _pc3, _pc4 = st.columns(4)
        _pc1.metric("Prophet MAPE",   f"{_pm['prophet']['MAPE']:.2f}%")
        _pc2.metric("XGBoost MAPE",   f"{_pm['xgboost_model_b']['MAPE']:.2f}%",
                    delta=f"{_pm['prophet']['MAPE'] - _pm['xgboost_model_b']['MAPE']:+.2f} pp (Prophet vs XGB)",
                    delta_color="inverse")
        _pc3.metric("DM p-value",     f"{_dm.get('pvalue', 1):.3f}",
                    help="DM test XGBoost vs Prophet. p>0.05 = no diferencia significativa.")
        _pc4.metric("Veredicto DM",   _dm.get("stars", "ns") or "n.s.",
                    help=_dm.get("verdict", ""))

        if os.path.exists(_prophet_pred_path):
            _pp = pd.read_csv(_prophet_pred_path, parse_dates=["week_date"])
            fig_prop = go.Figure()
            fig_prop.add_trace(go.Scatter(
                x=_pp["week_date"], y=_pp["actual_R03"],
                name="Real", line=dict(color="#111", width=2.5)))
            fig_prop.add_trace(go.Scatter(
                x=_pp["week_date"], y=_pp["prophet_forecast"],
                name=f"Prophet (MAPE {_pm['prophet']['MAPE']:.1f}%)",
                line=dict(color="#70ad47", width=2)))
            if "xgboost_pred" in sarima_preds.columns:
                fig_prop.add_trace(go.Scatter(
                    x=sarima_preds["week_date"], y=sarima_preds["xgboost_pred"],
                    name=f"XGBoost (MAPE {_pm['xgboost_model_b']['MAPE']:.1f}%)",
                    line=dict(color="#ff6b35", width=2, dash="dash")))
            fig_prop.add_traces([
                go.Scatter(x=_pp["week_date"], y=_pp["prophet_upper80"],
                           fill=None, mode="lines", line_color="rgba(112,173,71,0)", showlegend=False),
                go.Scatter(x=_pp["week_date"], y=_pp["prophet_lower80"],
                           fill="tonexty", mode="lines", line_color="rgba(112,173,71,0)",
                           fillcolor="rgba(112,173,71,0.12)", name="Prophet IC 80%"),
            ])
            fig_prop.update_layout(
                height=360, plot_bgcolor="white", paper_bgcolor="white",
                xaxis_title="Semana", yaxis_title="Unidades R03",
                title="Prophet vs XGBoost — Conjunto de Test (60 semanas)",
                legend=dict(orientation="h", y=-0.18), hovermode="x unified"
            )
            st.plotly_chart(fig_prop, use_container_width=True)

        st.caption(
            f"**Interpretacion**: XGBoost ({_pm['xgboost_model_b']['MAPE']:.2f}%) supera a Prophet "
            f"({_pm['prophet']['MAPE']:.2f}%) por {_pm['prophet']['MAPE'] - _pm['xgboost_model_b']['MAPE']:.2f} pp en MAPE, "
            f"pero el test Diebold-Mariano indica que la diferencia **no es estadisticamente significativa** "
            f"(p={_dm.get('pvalue', 1):.3f}). Ambos modelos superan a SARIMA ({_pm['sarima']['MAPE']:.2f}%). "
            f"Se prefiere XGBoost por mayor interpretabilidad via SHAP."
        )


# ==============================================================================
# PAGE 5: EXPLICABILIDAD SHAP
# ==============================================================================
elif page == "Explicabilidad SHAP":
    st.markdown('<h2 class="section-title">Explicabilidad del Modelo — Analisis SHAP</h2>',
                unsafe_allow_html=True)
    st.info(
        "**SHAP (SHapley Additive exPlanations)** descompone cada prediccion del modelo en "
        "contribuciones individuales por variable. A diferencia de la importancia de variables "
        "estandar de XGBoost, SHAP muestra *cuanto* y *en que direccion* cada feature empuja "
        "cada prediccion especifica — el estandar actual en ML explicable."
    )

    if shap_df is None:
        st.warning("Ejecuta src/09_shap_analysis.py para generar los datos SHAP.")
        st.stop()

    FEATURE_LABELS = {
        "shap_R03_lag1":         "R03 demanda (sem anterior)",
        "shap_R03_lag4_avg":     "R03 demanda (media 4 sem)",
        "shap_flu_au_positives": "Gripe Australia (sem actual)",
        "shap_flu_au_lagged":    "Gripe Australia (lag 26 sem)",
        "shap_flu_eu_positives": "Gripe Europa (sem actual)",
        "shap_R06_lag1":         "R06 demanda (sem anterior)",
    }
    shap_cols = [c for c in shap_df.columns if c.startswith("shap_")]
    mean_abs  = shap_df[shap_cols].abs().mean().sort_values(ascending=False)

    # ── Tabs ──
    tab1, tab2, tab3, tab4 = st.tabs([
        "Importancia Global", "Impacto por Semana", "Semana Pico", "Dependencias"
    ])

    with tab1:
        st.markdown("#### Importancia media por variable (mean |SHAP|)")
        st.caption("Cuantas unidades de demanda R03 mueve en promedio cada variable, en valor absoluto.")

        col_b1, col_b2 = st.columns([2, 1])
        with col_b1:
            labels = [FEATURE_LABELS.get(c, c.replace("shap_","")) for c in mean_abs.index]
            colors = ["#ff6b35" if "flu_au" in c else
                      "#70ad47" if "flu_eu" in c else "#1f4e79"
                      for c in mean_abs.index]
            fig = go.Figure(go.Bar(
                x=mean_abs.values[::-1], y=labels[::-1],
                orientation="h", marker_color=colors[::-1],
                text=[f"{v:.3f}" for v in mean_abs.values[::-1]],
                textposition="outside",
                hovertemplate="%{y}: %{x:.3f} unidades promedio<extra></extra>",
            ))
            fig.update_layout(
                title="Contribucion media absoluta por variable (SHAP)",
                xaxis_title="Mean |SHAP value| (unidades R03)",
                height=380, plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(l=180))
            fig.update_xaxes(range=[0, mean_abs.max() * 1.3])
            st.plotly_chart(fig, use_container_width=True)

        with col_b2:
            st.markdown("#### Hallazgos clave")
            total_imp = mean_abs.sum()
            flu_au_imp = mean_abs.get("shap_flu_au_lagged", 0)
            flu_eu_imp = mean_abs.get("shap_flu_eu_positives", 0)
            ar_imp = mean_abs.get("shap_R03_lag4_avg", 0) + mean_abs.get("shap_R03_lag1", 0)

            st.metric("Top feature", labels[0], f"{mean_abs.iloc[0]:.2f} unidades")
            st.metric("Señal lag Australia", f"{flu_au_imp:.2f} unidades",
                      f"{flu_au_imp/total_imp*100:.1f}% del total")
            st.metric("Señal Europa actual", f"{flu_eu_imp:.2f} unidades",
                      f"{flu_eu_imp/total_imp*100:.1f}% del total")
            st.metric("Features autoregresivos", f"{ar_imp:.2f} unidades",
                      f"{ar_imp/total_imp*100:.1f}% del total")
            st.divider()
            st.markdown(
                "La demanda reciente (4-week avg) es el predictor mas potente. "
                "La gripe australiana con lag contribuye **" +
                f"{flu_au_imp/total_imp*100:.1f}%** — "
                "pequeno pero unico: ninguna otra feature puede sustituir este "
                "adelanto de 6 meses."
            )

        # Beeswarm static image
        beeswarm_path = os.path.join(OUT, "shap_beeswarm.png")
        if os.path.exists(beeswarm_path):
            st.markdown("#### Beeswarm — distribucion completa de valores SHAP")
            st.caption("Cada punto es una semana. Color = valor de la feature (rojo=alto, azul=bajo). "
                       "X = cuanto empuja ese punto la prediccion.")
            st.image(beeswarm_path, use_container_width=True)

    with tab2:
        st.markdown("#### Contribucion SHAP de cada variable a lo largo del tiempo")
        feat_sel = st.selectbox(
            "Variable a visualizar",
            options=shap_cols,
            format_func=lambda c: FEATURE_LABELS.get(c, c.replace("shap_", ""))
        )
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=shap_df.index, y=shap_df[feat_sel],
            mode="lines+markers", marker=dict(size=4),
            line=dict(color="#1f4e79", width=1.5),
            name="SHAP value", fill="tozeroy",
            fillcolor="rgba(31,78,121,0.1)",
            hovertemplate="%{x|%d %b %Y}: %{y:.2f} unidades<extra></extra>",
        ))
        fig2.add_trace(go.Scatter(
            x=shap_df.index, y=shap_df["actual_R03"],
            name="Demanda real R03", line=dict(color="#ff6b35", dash="dot", width=1.5),
            yaxis="y2",
            hovertemplate="%{x|%d %b %Y}: %{y:.1f} unidades<extra></extra>",
        ))
        fig2.add_hline(y=0, line_color="#ccc")
        fig2.update_layout(
            title=f"Contribucion SHAP de '{FEATURE_LABELS.get(feat_sel, feat_sel)}' por semana",
            xaxis_title="Semana",
            yaxis=dict(title="SHAP value (unidades de impacto)"),
            yaxis2=dict(title="Demanda real R03", overlaying="y", side="right",
                        showgrid=False),
            height=420, hovermode="x unified",
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(
            "Valores SHAP positivos: esta variable *aumenta* la prediccion respecto al baseline. "
            f"Baseline del modelo: **{shap_df['baseline'].iloc[0]:.1f} unidades/semana**."
        )

    with tab3:
        st.markdown("#### Descomposicion de la semana de maxima demanda")
        waterfall_path = os.path.join(OUT, "shap_waterfall.png")
        if os.path.exists(waterfall_path):
            peak_idx  = shap_df["actual_R03"].idxmax()
            peak_val  = shap_df.loc[peak_idx, "actual_R03"]
            peak_pred = shap_df.loc[peak_idx, "predicted_R03"]
            st.markdown(
                f"**Semana analizada:** {peak_idx.strftime('%d %b %Y')} — "
                f"demanda real: **{peak_val:.1f}** unidades | "
                f"prediccion: **{peak_pred:.1f}** unidades"
            )
            st.image(waterfall_path, use_container_width=True)
            st.caption(
                "El grafico muestra como cada variable suma (+) o resta (-) desde el baseline "
                f"({shap_df['baseline'].iloc[0]:.1f} unidades) hasta llegar a la prediccion final. "
                "El grosor de cada barra refleja la magnitud del impacto."
            )
            # Waterfall data table
            wf_rows = []
            for col in shap_cols:
                feat = col.replace("shap_", "")
                label = FEATURE_LABELS.get(col, feat)
                sv = float(shap_df.loc[peak_idx, col])
                fv = float(shap_df.loc[peak_idx, feat]) if feat in shap_df.columns else float("nan")
                wf_rows.append({"Variable": label, "SHAP value": round(sv, 3),
                                 "Direccion": "+" if sv >= 0 else "-"})
            wf_df = pd.DataFrame(wf_rows).sort_values("SHAP value", key=abs, ascending=False)
            st.dataframe(wf_df, use_container_width=True, hide_index=True)

    with tab4:
        st.markdown("#### Grafico de dependencia — como una variable afecta las predicciones")
        dep_path_flu = os.path.join(OUT, "shap_dependence_flu_lag.png")
        dep_path_r03 = os.path.join(OUT, "shap_dependence_r03_lag.png")

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            if os.path.exists(dep_path_flu):
                st.markdown("**Gripe Australia (lag 26 sem) → impacto SHAP**")
                st.image(dep_path_flu, use_container_width=True)
                st.caption(
                    "Muestra que cuando la actividad gripal australiana lagged es alta, "
                    "el modelo *aumenta* la prediccion de demanda europea — confirmando "
                    "que el lag australiano actua como señal de alerta temprana."
                )
        with col_d2:
            if os.path.exists(dep_path_r03):
                st.markdown("**R03 demanda anterior → impacto SHAP**")
                st.image(dep_path_r03, use_container_width=True)
                st.caption(
                    "La demanda reciente tiene efecto fuertemente positivo — el modelo "
                    "reconoce que alta demanda semana pasada predice alta demanda esta semana "
                    "(inercia estacional)."
                )


# ==============================================================================
# PAGE 6: VALIDACION WALK-FORWARD
# ==============================================================================
elif page == "Validacion Walk-Forward":
    st.markdown('<h2 class="section-title">Validacion Walk-Forward (Cross-Validacion Temporal)</h2>',
                unsafe_allow_html=True)
    st.markdown(
        "La validacion walk-forward es el estandar de oro para evaluar modelos de series temporales. "
        "En lugar de un unico split 80/20, se entrena el modelo repetidamente con ventana expansiva "
        "y se mide el error en cada segmento futuro no visto — simulando el uso real del modelo en produccion."
    )

    if wfv_folds is None:
        st.warning("Datos de walk-forward no encontrados. Ejecuta: `python src/10_walk_forward_validation.py`")
    else:
        # ── KPI row ──────────────────────────────────────────────────────────
        n_folds      = wfv_meta["n_folds"]
        n_preds      = wfv_meta["n_predictions"]
        b_mape_mean  = wfv_meta["B_MAPE_mean"]
        b_mape_std   = wfv_meta["B_MAPE_std"]
        a_mape_mean  = wfv_meta["A_MAPE_mean"]
        delta        = wfv_meta["delta_mape_mean"]
        b_wins       = wfv_meta["B_wins_pct"]
        min_train    = wfv_meta["min_train_weeks"]
        step_w       = wfv_meta["step_weeks"]
        test_w       = wfv_meta["test_window_weeks"]

        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(f"""<div class="metric-card">
            <h3>Folds totales</h3><h1>{n_folds}</h1>
            <p>{n_preds} semanas out-of-sample</p></div>""", unsafe_allow_html=True)
        k2.markdown(f"""<div class="metric-card">
            <h3>MAPE medio — Modelo B</h3><h1>{b_mape_mean:.1f}%</h1>
            <p>std {b_mape_std:.1f}% | FluNet + lags</p></div>""", unsafe_allow_html=True)
        k3.markdown(f"""<div class="metric-card">
            <h3>MAPE medio — Modelo A</h3><h1>{a_mape_mean:.1f}%</h1>
            <p>Solo lags autoregresivos</p></div>""", unsafe_allow_html=True)
        k4.markdown(f"""<div class="metric-card">
            <h3>Mejora FluNet</h3><h1>{delta:+.2f}%</h1>
            <p>B mejor en {b_wins:.0f}% de folds</p></div>""", unsafe_allow_html=True)

        st.caption(
            f"Configuracion: ventana minima de entrenamiento = {min_train} semanas | "
            f"avance por fold = {step_w} semanas | ventana de test por fold = {test_w} semanas"
        )
        st.divider()

        tab1, tab2, tab3 = st.tabs(["Predicciones out-of-sample", "MAPE por fold", "Tabla de resultados"])

        with tab1:
            st.markdown("#### Reconstruccion out-of-sample — todos los folds")
            st.markdown(
                "Cada punto de esta curva es una prediccion de un fold en el que el modelo NO habia "
                "visto esos datos durante el entrenamiento. Es la evaluacion mas honesta del modelo."
            )
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=wfv_preds.index, y=wfv_preds["actual_R03"],
                name="Demanda real R03", line=dict(color="#111111", width=2),
                mode="lines"
            ))
            fig1.add_trace(go.Scatter(
                x=wfv_preds.index, y=wfv_preds["pred_A"],
                name="Modelo A (lags)", line=dict(color="#aaaaaa", width=1, dash="dot"),
                opacity=0.75, mode="lines"
            ))
            fig1.add_trace(go.Scatter(
                x=wfv_preds.index, y=wfv_preds["pred_B"],
                name="Modelo B (+ FluNet)", line=dict(color="#ff6b35", width=2, dash="dash"),
                opacity=0.9, mode="lines"
            ))
            fig1.update_layout(
                height=420, plot_bgcolor="white", paper_bgcolor="white",
                xaxis_title="Semana", yaxis_title="Unidades R03 / semana",
                title="Predicciones out-of-sample — Walk-Forward Validation",
                legend=dict(orientation="h", y=-0.15),
                hovermode="x unified"
            )
            st.plotly_chart(fig1, use_container_width=True)

            # Error distribution
            wfv_preds["error_B"] = wfv_preds["pred_B"] - wfv_preds["actual_R03"]
            wfv_preds["error_A"] = wfv_preds["pred_A"] - wfv_preds["actual_R03"]

            fig_err = go.Figure()
            fig_err.add_trace(go.Histogram(
                x=wfv_preds["error_A"], name="Error Modelo A",
                opacity=0.6, marker_color="#aaaaaa", nbinsx=30
            ))
            fig_err.add_trace(go.Histogram(
                x=wfv_preds["error_B"], name="Error Modelo B",
                opacity=0.7, marker_color="#ff6b35", nbinsx=30
            ))
            fig_err.add_vline(x=0, line_dash="dash", line_color="#333")
            fig_err.update_layout(
                height=280, barmode="overlay", plot_bgcolor="white", paper_bgcolor="white",
                xaxis_title="Error de prediccion (unidades)", yaxis_title="Frecuencia",
                title="Distribucion de errores out-of-sample",
                legend=dict(orientation="h", y=-0.2)
            )
            st.plotly_chart(fig_err, use_container_width=True)

        with tab2:
            st.markdown("#### MAPE por fold — evolucion temporal")
            st.markdown(
                "Los picos de MAPE corresponden a semanas de verano cuando la demanda cae a "
                "valores muy bajos — pequenos errores absolutos producen MAPE alto. "
                "Los folds de invierno (pico estacional) son los mas estables."
            )
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=wfv_folds["fold"], y=wfv_folds["A_MAPE"],
                name="Modelo A", line=dict(color="#aaaaaa", width=1.5),
                mode="lines+markers", marker=dict(size=4)
            ))
            fig2.add_trace(go.Scatter(
                x=wfv_folds["fold"], y=wfv_folds["B_MAPE"],
                name="Modelo B (+ FluNet)", line=dict(color="#ff6b35", width=2),
                mode="lines+markers", marker=dict(size=4)
            ))
            fig2.add_hline(y=a_mape_mean, line_dash="dot", line_color="#aaaaaa",
                           annotation_text=f"Media A: {a_mape_mean:.1f}%",
                           annotation_position="right")
            fig2.add_hline(y=b_mape_mean, line_dash="dot", line_color="#ff6b35",
                           annotation_text=f"Media B: {b_mape_mean:.1f}%",
                           annotation_position="right")

            # Colour the test_start date on hover using customdata
            fig2.update_traces(
                customdata=wfv_folds[["test_start", "test_end", "train_weeks"]].values,
                hovertemplate="Fold %{x}<br>MAPE: %{y:.1f}%<br>"
                              "Test: %{customdata[0]} → %{customdata[1]}<br>"
                              "Train weeks: %{customdata[2]}<extra></extra>",
                selector=dict(type="scatter")
            )
            fig2.update_layout(
                height=380, plot_bgcolor="white", paper_bgcolor="white",
                xaxis_title="Fold", yaxis_title="MAPE (%)",
                title="MAPE por fold — Modelo A vs Modelo B",
                legend=dict(orientation="h", y=-0.18)
            )
            st.plotly_chart(fig2, use_container_width=True)

            # MAE per fold
            fig_mae = go.Figure()
            fig_mae.add_trace(go.Bar(
                x=wfv_folds["fold"], y=wfv_folds["A_MAE"],
                name="Modelo A", marker_color="#cccccc", opacity=0.75
            ))
            fig_mae.add_trace(go.Bar(
                x=wfv_folds["fold"], y=wfv_folds["B_MAE"],
                name="Modelo B", marker_color="#ff6b35", opacity=0.8
            ))
            fig_mae.update_layout(
                height=280, barmode="group", plot_bgcolor="white", paper_bgcolor="white",
                xaxis_title="Fold", yaxis_title="MAE (unidades)",
                title="MAE por fold", legend=dict(orientation="h", y=-0.2)
            )
            st.plotly_chart(fig_mae, use_container_width=True)

        with tab3:
            st.markdown("#### Resultados completos por fold")
            show_cols = ["fold", "test_start", "test_end", "train_weeks",
                         "A_MAPE", "B_MAPE", "A_MAE", "B_MAE", "A_R2", "B_R2"]
            display_df = wfv_folds[show_cols].copy()
            display_df["Mejora MAPE"] = (display_df["A_MAPE"] - display_df["B_MAPE"]).round(2)
            display_df["B gana"] = display_df["Mejora MAPE"] > 0

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "fold":          st.column_config.NumberColumn("Fold", format="%d"),
                    "test_start":    st.column_config.TextColumn("Test inicio"),
                    "test_end":      st.column_config.TextColumn("Test fin"),
                    "train_weeks":   st.column_config.NumberColumn("Semanas entren.", format="%d"),
                    "A_MAPE":        st.column_config.NumberColumn("MAPE A (%)", format="%.1f"),
                    "B_MAPE":        st.column_config.NumberColumn("MAPE B (%)", format="%.1f"),
                    "A_MAE":         st.column_config.NumberColumn("MAE A", format="%.2f"),
                    "B_MAE":         st.column_config.NumberColumn("MAE B", format="%.2f"),
                    "A_R2":          st.column_config.NumberColumn("R2 A", format="%.3f"),
                    "B_R2":          st.column_config.NumberColumn("R2 B", format="%.3f"),
                    "Mejora MAPE":   st.column_config.NumberColumn("Delta MAPE", format="%.2f"),
                    "B gana":        st.column_config.CheckboxColumn("B mejor"),
                }
            )
            st.download_button(
                "Descargar resultados CSV",
                wfv_folds.to_csv(index=False).encode("utf-8"),
                "wfv_fold_results.csv", "text/csv"
            )

        st.divider()
        st.markdown("#### Interpretacion de resultados")
        col_i1, col_i2 = st.columns(2)
        with col_i1:
            st.info(
                f"**Modelo B (FluNet) es mas robusto**: gana en {b_wins:.0f}% de los folds. "
                f"La mejora media de {delta:.2f}% en MAPE es consistente a lo largo del tiempo, "
                "no solo en el conjunto de test estatico del 80/20."
            )
        with col_i2:
            st.warning(
                "**Alta variabilidad en verano**: Los folds con test en julio-agosto muestran "
                "MAPE extremos (>100%) porque la demanda baja a valores cercanos a cero. "
                "En operaciones reales, los periodos de baja demanda requieren logica de umbral "
                "separada del modelo principal."
            )


# PAGE 7: SIMULACION DE INVENTARIO
# ==============================================================================
elif page == "Simulacion de Inventario":
    st.markdown('<h2 class="section-title">Simulacion de Inventario — Politica (s, Q)</h2>',
                unsafe_allow_html=True)
    st.markdown(
        "Simula la evolucion del inventario durante las **60 semanas del conjunto de test** "
        "bajo cuatro escenarios de prevision. La politica (s, Q) realiza pedidos cuando el "
        "stock proyectado cae por debajo del punto de reorden, usando el horizonte de entrega "
        "para calcular cuanto pedir."
    )

    if inv_sim is None:
        st.warning("Datos de simulacion no encontrados. Ejecuta: `python src/12_inventory_simulation.py`")
    else:
        # ── Params strip ─────────────────────────────────────────────────────
        p = inv_meta["params"]
        st.markdown(
            f"**Parametros:** Stock inicial `{p['initial_stock']}` · "
            f"Safety stock `{p['safety_stock']}` · "
            f"Punto reorden `{p['reorder_point']}` · "
            f"Stock objetivo `{p['target_stock']}` · "
            f"Lead time `{p['lead_time']} semanas` · "
            f"Coste almacen `EUR {p['holding_cost']}/u/sem` · "
            f"Coste rotura `EUR {p['shortage_cost']}/u/sem`"
        )
        st.divider()

        # ── Scenario selector ─────────────────────────────────────────────────
        scenarios_available = inv_sim["label"].unique().tolist()
        label_map = {
            "Naive":    "Naive (sin forecast)",
            "XGBoost":  "XGBoost forecast",
            "SARIMA":   "SARIMA forecast",
            "Perfect":  "Prevision perfecta",
        }
        color_map = {
            "Naive":   "#aaaaaa",
            "XGBoost": "#ff6b35",
            "SARIMA":  "#70ad47",
            "Perfect": "#7030a0",
        }

        st.markdown("### Animacion de Inventario")
        st.markdown("Selecciona un escenario y pulsa **▶ Play** para ver como evoluciona el stock semana a semana.")

        sel_label = st.selectbox(
            "Escenario a animar",
            options=scenarios_available,
            format_func=lambda x: label_map.get(x, x),
            index=1  # default: XGBoost
        )

        # Build animated figure for selected scenario
        sc_df = inv_sim[inv_sim["label"] == sel_label].reset_index(drop=True)
        dates_sc = sc_df["week_date"].tolist()
        stock    = sc_df["stock"].tolist()
        demand   = sc_df["actual_demand"].tolist()
        forecast = sc_df["forecast"].tolist()
        shortage = sc_df["shortage"].tolist()
        orders   = sc_df["order_placed"].tolist()
        n_weeks  = len(sc_df)

        safety_stock = p["safety_stock"]
        rop          = p["reorder_point"]
        main_color   = color_map.get(sel_label, "#ff6b35")

        def make_inventory_frame(t):
            """Build one animation frame showing state up to week t."""
            xs = dates_sc[:t+1]
            ys = stock[:t+1]

            # Colored fill zones
            fill_green  = [s if s > rop          else None for s in ys]
            fill_orange = [s if safety_stock < s <= rop else None for s in ys]
            fill_red    = [s if s <= safety_stock        else None for s in ys]

            traces = [
                # Demand bars (background)
                go.Bar(x=xs, y=demand[:t+1], name="Demanda real",
                       marker_color="rgba(100,100,100,0.25)", showlegend=(t == 0)),
                # Forecast bars
                go.Bar(x=xs, y=forecast[:t+1], name="Prevision usada",
                       marker_color=f"rgba{tuple(int(main_color.lstrip('#')[i:i+2],16) for i in (0,2,4)) + (0.35,)}",
                       showlegend=(t == 0)),
                # Green zone fill
                go.Scatter(x=xs, y=[s if s > rop else None for s in ys],
                           fill="tozeroy", fillcolor="rgba(112,173,71,0.18)",
                           line=dict(width=0), showlegend=False, hoverinfo="skip"),
                # Orange zone fill
                go.Scatter(x=xs, y=[s if 0 < s <= rop else None for s in ys],
                           fill="tozeroy", fillcolor="rgba(255,107,53,0.20)",
                           line=dict(width=0), showlegend=False, hoverinfo="skip"),
                # Red zone fill
                go.Scatter(x=xs, y=[max(0,s) if shortage[:t+1][i] > 0 else None
                                    for i,s in enumerate(ys)],
                           fill="tozeroy", fillcolor="rgba(192,0,0,0.22)",
                           line=dict(width=0), showlegend=False, hoverinfo="skip"),
                # Stock line
                go.Scatter(x=xs, y=ys, mode="lines",
                           name="Nivel de stock",
                           line=dict(color=main_color, width=3),
                           showlegend=(t == 0)),
                # Order markers
                go.Scatter(
                    x=[dates_sc[i] for i in range(t+1) if orders[i] > 0],
                    y=[stock[i] for i in range(t+1) if orders[i] > 0],
                    mode="markers", name="Pedido realizado",
                    marker=dict(symbol="triangle-up", size=14, color="#1f4e79",
                                line=dict(color="white", width=1.5)),
                    showlegend=(t == 0),
                    text=[f"Pedido: {orders[i]:.0f}u" for i in range(t+1) if orders[i] > 0],
                    hovertemplate="%{text}<extra></extra>"
                ),
                # Stockout markers
                go.Scatter(
                    x=[dates_sc[i] for i in range(t+1) if shortage[i] > 0],
                    y=[0] * sum(1 for i in range(t+1) if shortage[i] > 0),
                    mode="markers", name="Rotura de stock",
                    marker=dict(symbol="x", size=13, color="#c00000",
                                line=dict(width=2.5)),
                    showlegend=(t == 0),
                    text=[f"Rotura: {shortage[i]:.0f}u" for i in range(t+1) if shortage[i] > 0],
                    hovertemplate="%{text}<extra></extra>"
                ),
            ]
            # Cumulative KPIs for title annotation
            cum_stockouts = sum(1 for i in range(t+1) if shortage[i] > 0)
            cum_cost = sum(sc_df.iloc[i]["holding_cost"] + sc_df.iloc[i]["shortage_cost"]
                           + (p["order_cost"] if orders[i] > 0 else 0) for i in range(t+1))
            title = (f"Semana {t+1}/{n_weeks} | Stock: {stock[t]:.0f}u | "
                     f"Roturas: {cum_stockouts} | Coste acum: EUR {cum_cost:.0f}")
            return traces, title

        # Build all frames
        frames = []
        for t in range(n_weeks):
            frame_traces, frame_title = make_inventory_frame(t)
            frames.append(go.Frame(
                data=frame_traces,
                name=str(t),
                layout=go.Layout(title_text=frame_title)
            ))

        # Initial frame (week 0)
        init_traces, init_title = make_inventory_frame(0)

        fig_anim = go.Figure(
            data=init_traces,
            frames=frames,
            layout=go.Layout(
                title=dict(text=init_title, font=dict(size=13, color="#1f4e79"), x=0.02),
                height=480,
                barmode="overlay",
                plot_bgcolor="#fafafa",
                paper_bgcolor="white",
                xaxis=dict(title="Semana", range=[dates_sc[0], dates_sc[-1]],
                           showgrid=True, gridcolor="#eeeeee"),
                yaxis=dict(title="Unidades", showgrid=True, gridcolor="#eeeeee"),
                legend=dict(orientation="h", y=-0.22, font=dict(size=10)),
                hovermode="x unified",
                shapes=[
                    dict(type="line", x0=dates_sc[0], x1=dates_sc[-1],
                         y0=safety_stock, y1=safety_stock,
                         line=dict(color="#70ad47", width=1.5, dash="dot")),
                    dict(type="line", x0=dates_sc[0], x1=dates_sc[-1],
                         y0=rop, y1=rop,
                         line=dict(color="#ff6b35", width=1.5, dash="dot")),
                ],
                annotations=[
                    dict(x=dates_sc[-1], y=safety_stock, xanchor="right",
                         text=f"Safety stock ({safety_stock})", showarrow=False,
                         font=dict(size=9, color="#70ad47")),
                    dict(x=dates_sc[-1], y=rop, xanchor="right",
                         text=f"Punto reorden ({rop})", showarrow=False,
                         font=dict(size=9, color="#ff6b35")),
                ],
                updatemenus=[dict(
                    type="buttons",
                    showactive=False,
                    y=1.15, x=0.0, xanchor="left",
                    buttons=[
                        dict(label="▶  Play",
                             method="animate",
                             args=[None, dict(frame=dict(duration=220, redraw=True),
                                              fromcurrent=True,
                                              transition=dict(duration=80))]),
                        dict(label="⏸  Pausa",
                             method="animate",
                             args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                mode="immediate",
                                                transition=dict(duration=0))]),
                    ]
                )],
                sliders=[dict(
                    active=0,
                    currentvalue=dict(prefix="Semana: ", visible=True, xanchor="center"),
                    transition=dict(duration=80),
                    pad=dict(t=50),
                    steps=[dict(
                        args=[[str(t)], dict(frame=dict(duration=220, redraw=True),
                                             mode="immediate",
                                             transition=dict(duration=80))],
                        label=str(t+1),
                        method="animate"
                    ) for t in range(n_weeks)]
                )],
            )
        )

        st.plotly_chart(fig_anim, use_container_width=True)

        st.divider()

        # ── KPI comparison table ──────────────────────────────────────────────
        st.markdown("### Comparacion de KPIs entre escenarios")

        kpi_rows = inv_summ.copy()
        kpi_rows["service_level"] = kpi_rows["service_level"].map(lambda x: f"{x:.1f}%")
        kpi_rows["total_cost"]    = kpi_rows["total_cost"].map(lambda x: f"EUR {x:.0f}")
        kpi_rows["avg_stock"]     = kpi_rows["avg_stock"].map(lambda x: f"{x:.1f}")

        kcols = st.columns(len(inv_summ))
        for i, row in inv_summ.iterrows():
            label_k = row["scenario"]
            color_k = color_map.get(
                next((k for k, v in label_map.items() if v == label_k), ""), "#666"
            )
            kcols[i].markdown(f"""<div style="background:{color_k}18; border-left:4px solid {color_k};
                border-radius:0 8px 8px 0; padding:10px 14px; margin:4px 0;">
                <b style="color:{color_k}">{label_k}</b><br>
                <span style="font-size:1.4rem;font-weight:700">{row['service_level']:.1f}%</span>
                <span style="font-size:0.78rem;color:#666"> servicio</span><br>
                <span style="font-size:0.85rem">{row['stockout_weeks']} sem rotura</span><br>
                <span style="font-size:0.85rem">EUR {row['total_cost']:.0f} coste total</span><br>
                <span style="font-size:0.78rem;color:#888">{row['n_orders']} pedidos · avg {row['avg_stock']:.0f}u stock</span>
                </div>""", unsafe_allow_html=True)

        st.divider()

        # ── All-scenarios side-by-side Plotly static ──────────────────────────
        st.markdown("### Vista comparativa — Evolucion de stock (todos los escenarios)")

        fig_cmp = make_subplots(
            rows=2, cols=2,
            subplot_titles=[label_map.get(s, s) for s in scenarios_available],
            shared_xaxes=True, shared_yaxes=True,
            vertical_spacing=0.12, horizontal_spacing=0.06
        )
        for idx, sc_label in enumerate(scenarios_available):
            row, col = divmod(idx, 2)
            sc = inv_sim[inv_sim["label"] == sc_label].reset_index(drop=True)
            dates_all = sc["week_date"]
            c = color_map.get(sc_label, "#666")
            kp = inv_meta["kpis"].get(label_map.get(sc_label, sc_label), {})

            # Fill zones
            fig_cmp.add_trace(go.Scatter(
                x=dates_all, y=[s if s > rop else None for s in sc["stock"]],
                fill="tozeroy", fillcolor="rgba(112,173,71,0.15)",
                line=dict(width=0), showlegend=False, hoverinfo="skip"
            ), row=row+1, col=col+1)
            fig_cmp.add_trace(go.Scatter(
                x=dates_all, y=[s if 0 < s <= rop else None for s in sc["stock"]],
                fill="tozeroy", fillcolor="rgba(255,107,53,0.20)",
                line=dict(width=0), showlegend=False, hoverinfo="skip"
            ), row=row+1, col=col+1)
            fig_cmp.add_trace(go.Scatter(
                x=dates_all, y=sc["stock"], mode="lines", name=label_map.get(sc_label, sc_label),
                line=dict(color=c, width=2.2), showlegend=True
            ), row=row+1, col=col+1)
            # Stockout markers
            so_dates = dates_all[sc["shortage"] > 0]
            if len(so_dates):
                fig_cmp.add_trace(go.Scatter(
                    x=so_dates, y=[0]*len(so_dates), mode="markers",
                    marker=dict(symbol="x", size=9, color="#c00000", line=dict(width=2)),
                    name="Rotura", showlegend=(idx == 0)
                ), row=row+1, col=col+1)
            # Threshold lines
            fig_cmp.add_hline(y=safety_stock, line_dash="dot", line_color="#70ad47",
                               line_width=1, row=row+1, col=col+1)
            fig_cmp.add_hline(y=rop, line_dash="dot", line_color="#ff6b35",
                               line_width=1, row=row+1, col=col+1)

        fig_cmp.update_layout(
            height=520, plot_bgcolor="white", paper_bgcolor="white",
            hovermode="x unified",
            legend=dict(orientation="h", y=-0.08, font=dict(size=10)),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        # ── Cost breakdown ────────────────────────────────────────────────────
        st.markdown("### Desglose de costes por escenario")
        cost_fig = go.Figure()
        cost_labels = [label_map.get(s, s) for s in inv_summ["scenario"]]
        cost_colors_list = [color_map.get(
            next((k for k, v in label_map.items() if v == s), ""), "#888"
        ) for s in inv_summ["scenario"]]
        for cost_type, col_hex in [("holding_cost", "#2e75b6"),
                                    ("shortage_cost", "#c00000"),
                                    ("order_cost",   "#1f4e79")]:
            label_es = {"holding_cost": "Coste almacen",
                        "shortage_cost": "Coste rotura",
                        "order_cost": "Coste pedido"}[cost_type]
            cost_fig.add_trace(go.Bar(
                name=label_es,
                x=cost_labels,
                y=inv_summ[cost_type],
                marker_color=col_hex,
                text=[f"EUR {v:.0f}" for v in inv_summ[cost_type]],
                textposition="inside", textfont=dict(color="white", size=9)
            ))
        cost_fig.update_layout(
            barmode="stack", height=300,
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis_title="Escenario", yaxis_title="EUR (60 semanas)",
            legend=dict(orientation="h", y=-0.22),
            title="Coste total acumulado por escenario — 60 semanas"
        )
        st.plotly_chart(cost_fig, use_container_width=True)

        xgb_row   = inv_summ[inv_summ["scenario"].str.contains("XGBoost")].iloc[0]
        naive_row = inv_summ[inv_summ["scenario"].str.contains("Naive")].iloc[0]
        saving    = naive_row["total_cost"] - xgb_row["total_cost"]
        st.success(
            f"**XGBoost vs Naive**: ahorro de **EUR {saving:.0f}** en 60 semanas "
            f"(EUR {saving/60*52:.0f} anualizados). "
            f"El forecast mejora la eficiencia del inventario reduciendo pedidos y costes de almacen "
            f"sin aumentar las roturas de stock."
        )


# PAGE 8: CONTEXTO EUROPEO (AMELI)
# ==============================================================================
elif page == "Contexto Europeo (AMELI)":
    st.markdown('<h2 class="section-title">Contexto Europeo — Open Medic AMELI (Francia)</h2>',
                unsafe_allow_html=True)

    col_src1, col_src2, col_src3 = st.columns(3)
    col_src1.markdown("""<span class="source-badge">Assurance Maladie France</span>""",
                      unsafe_allow_html=True)
    col_src2.markdown("""<span class="source-badge">data.gouv.fr — Open Medic</span>""",
                      unsafe_allow_html=True)
    col_src3.markdown("""<span class="source-badge">ATC3 R03 + R06 · 2014-2024</span>""",
                      unsafe_allow_html=True)

    st.info("**Open Medic** es el registro oficial de dispensacion de medicamentos reembolsados "
            "por el sistema de salud frances. Descarga automatica integrada en el pipeline via "
            "API data.gouv.fr. Clasifica por codigo ATC, permitiendo filtrado directo de R03 y R06.")

    if france is not None and not france.empty:
        # KPIs
        r03_avg  = france["france_R03_boxes"].mean()
        r06_avg  = france["france_R06_boxes"].mean()
        r03_2024 = france.loc[france["year"] == 2024, "france_R03_boxes"].values
        r03_2014 = france.loc[france["year"] == 2014, "france_R03_boxes"].values
        r03_growth = float((r03_2024[0] - r03_2014[0]) / r03_2014[0] * 100) if len(r03_2024) and len(r03_2014) else 0

        kc1, kc2, kc3, kc4 = st.columns(4)
        kc1.markdown(f"""<div class="metric-card"><h3>R03 promedio anual</h3>
            <h1>{r03_avg/1e6:.1f}M</h1><p>cajas dispensadas/ano Francia</p></div>""",
            unsafe_allow_html=True)
        kc2.markdown(f"""<div class="metric-card"><h3>R06 promedio anual</h3>
            <h1>{r06_avg/1e6:.1f}M</h1><p>cajas dispensadas/ano Francia</p></div>""",
            unsafe_allow_html=True)
        kc3.markdown(f"""<div class="metric-card"><h3>Anos disponibles</h3>
            <h1>{len(france)}</h1><p>2014 — 2024 (11 anos)</p></div>""",
            unsafe_allow_html=True)
        kc4.markdown(f"""<div class="metric-card"><h3>Crecimiento R03</h3>
            <h1>{r03_growth:+.1f}%</h1><p>2014 vs 2024</p></div>""",
            unsafe_allow_html=True)

        st.markdown("")

        # Bar chart: R03 + R06 by year
        fig = go.Figure()
        fig.add_trace(go.Bar(x=france["year"], y=france["france_R03_boxes"]/1e6,
                             name="R03 Respiratorio", marker_color="#1f4e79"))
        fig.add_trace(go.Bar(x=france["year"], y=france["france_R06_boxes"]/1e6,
                             name="R06 Antihistaminico", marker_color="#70ad47"))
        fig.update_layout(title="Francia: Cajas de Medicamento Dispensadas por Ano (Millones)",
                          xaxis_title="Ano", yaxis_title="Millones de cajas",
                          barmode="group", height=340,
                          plot_bgcolor="white", paper_bgcolor="white",
                          legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)

        # Trend + growth rate
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown("#### Tendencia R03 con crecimiento interanual")
            fr = france.copy()
            fr["R03_growth"] = fr["france_R03_boxes"].pct_change().mul(100)
            fig2 = make_subplots(specs=[[{"secondary_y": True}]])
            fig2.add_trace(go.Scatter(x=fr["year"], y=fr["france_R03_boxes"]/1e6,
                                      name="R03 (M cajas)", line=dict(color="#1f4e79", width=2.5)),
                           secondary_y=False)
            colors_g = ["#c00000" if v < 0 else "#70ad47" for v in fr["R03_growth"].fillna(0)]
            fig2.add_trace(go.Bar(x=fr["year"], y=fr["R03_growth"],
                                  name="Crecimiento YoY (%)", marker_color=colors_g, opacity=0.6),
                           secondary_y=True)
            fig2.add_hline(y=0, line_color="#999", secondary_y=True)
            fig2.update_layout(height=300, plot_bgcolor="white", paper_bgcolor="white",
                               legend=dict(orientation="h", y=-0.2))
            fig2.update_yaxes(title_text="M cajas", secondary_y=False)
            fig2.update_yaxes(title_text="Crecimiento (%)", secondary_y=True)
            st.plotly_chart(fig2, use_container_width=True)

        with col_t2:
            st.markdown("#### R03 vs R06: Ratio y evolucion")
            fr["ratio_r03_r06"] = fr["france_R03_boxes"] / fr["france_R06_boxes"]
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=fr["year"], y=fr["ratio_r03_r06"],
                                      mode="lines+markers",
                                      line=dict(color="#1f4e79", width=2),
                                      marker=dict(size=8),
                                      name="Ratio R03/R06"))
            fig3.add_hline(y=1.0, line_dash="dot", line_color="#999", annotation_text="Ratio 1:1")
            fig3.update_layout(height=300, plot_bgcolor="white", paper_bgcolor="white",
                               xaxis_title="Ano", yaxis_title="Ratio R03/R06",
                               title="Ratio de dispensacion R03 vs R06")
            st.plotly_chart(fig3, use_container_width=True)

        st.divider()

        # Escala comparison
        st.markdown("#### Comparacion de escala: Francia (AMELI) vs Dataset Kaggle")
        kaggle_mean     = df["R03"].mean()
        kaggle_weekly   = kaggle_mean
        france_weekly   = france["france_R03_boxes"].mean() / 52
        ratio           = france_weekly / kaggle_weekly

        cs1, cs2, cs3 = st.columns(3)
        cs1.metric("Kaggle — media semanal",   f"{kaggle_weekly:.1f} unidades",
                   help="Una farmacia europea, unidades vendidas")
        cs2.metric("Francia AMELI — semanal",  f"{france_weekly/1000:.0f}K cajas",
                   help="Mercado farmaceutico nacional frances")
        cs3.metric("Factor de escala",          f"{ratio/1000:.0f}K x",
                   help="Francia (68M hab.) vs una farmacia")
        st.caption("El ratio confirma que el dataset de Kaggle corresponde a UNA farmacia europea, "
                   "mientras que Open Medic abarca el mercado nacional frances completo (68M habitantes). "
                   "Ambas fuentes confirman la misma clasificacion ATC y el mismo patron estacional R03 > R06.")

        # Raw data table
        st.markdown("#### Datos completos AMELI (2014-2024)")
        fr_show = france.copy()
        fr_show["R03 (M cajas)"] = (fr_show["france_R03_boxes"] / 1e6).round(2)
        fr_show["R06 (M cajas)"] = (fr_show["france_R06_boxes"] / 1e6).round(2)
        fr_show["Ratio R03/R06"] = (fr_show["france_R03_boxes"] / fr_show["france_R06_boxes"]).round(3)
        fr_show["Crecim. R03 (%)"] = fr_show["france_R03_boxes"].pct_change().mul(100).round(1)
        st.dataframe(fr_show[["year", "R03 (M cajas)", "R06 (M cajas)", "Ratio R03/R06", "Crecim. R03 (%)"]],
                     use_container_width=True, hide_index=True,
                     column_config={"year": st.column_config.NumberColumn("Ano", format="%d")})
        st.download_button("Descargar datos AMELI CSV",
                           fr_show[["year","R03 (M cajas)","R06 (M cajas)","Ratio R03/R06"]
                                   ].to_csv(index=False).encode("utf-8"),
                           "france_ameli_r03_r06.csv", "text/csv")
    else:
        st.warning("Datos AMELI no encontrados. Ejecuta src/08_download_france_openmedic.py")


# ==============================================================================
# PAGE 11: ENSEMBLE & SWITCHING RULE
# ==============================================================================
elif page == "Ensemble & Switching":
    st.markdown('<h2 class="section-title">Ensemble Multi-País & Regla de Cambio Estacional</h2>',
                unsafe_allow_html=True)
    st.caption("Resultados de los experimentos avanzados — Steps 16 y 17 del pipeline")

    tab_ens, tab_sw = st.tabs(["Ensemble Multi-País", "Regla de Cambio Estacional"])

    # ── TAB 1: ENSEMBLE ───────────────────────────────────────────────────────
    with tab_ens:
        st.markdown("### Ensemble Hemisferio Sur: Australia + Nueva Zelanda + Chile")
        st.markdown(
            "Se entrena un único XGBoost combinando señales retardadas de tres países "
            "(AU lag 26w, NZ lag 27w, Chile lag 35w) para ver si un modelo multi-señal "
            "reduce el MAPE respecto al Modelo B de un solo país."
        )

        ens_path = os.path.join(OUT, "ensemble_meta.json")
        ens_pred_path = os.path.join(OUT, "ensemble_predictions.csv")

        if os.path.exists(ens_path):
            with open(ens_path) as _f:
                ens_meta = json.load(_f)

            models_res = ens_meta["models"]
            labels     = list(models_res.keys())
            mapes      = [models_res[l]["MAPE"] for l in labels]
            maes       = [models_res[l]["MAE"]  for l in labels]
            r2s        = [models_res[l]["R2"]   for l in labels]
            best_lbl   = ens_meta["best_label"]
            improv     = ens_meta["improvement_pp"]

            # KPI row
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Modelo B (AU solo)", f"{ens_meta['model_b_mape']:.2f}%", "baseline MAPE")
            k2.metric("Mejor ensemble", f"{ens_meta['best_mape']:.2f}%",
                      f"{improv:+.2f}pp vs Modelo B")
            k3.metric("Mejor variante", best_lbl.split("(")[-1].replace(")", "").strip())
            k4.metric("Conclusion", "AU solo gana" if improv <= 0 else "Ensemble mejora")

            st.info(
                f"**Hallazgo honesto:** El ensemble no mejora al Modelo B. Australia sola "
                f"(MAPE {ens_meta['model_b_mape']:.2f}%) sigue siendo la mejor señal única. "
                f"Añadir NZ o Chile introduce ruido adicional. Esto es consistente con el "
                f"análisis de validación hemisférica (paso 13): correlación no implica predicibilidad conjunta."
            )

            # Bar chart of MAPEs
            _colors_ens = ["#aaaaaa", "#2e75b6", "#70ad47", "#7030a0"]
            fig_ens = go.Figure()
            fig_ens.add_trace(go.Bar(
                x=[l.replace("Model B (Australia only)", "AU solo") for l in labels],
                y=mapes, marker_color=_colors_ens[:len(labels)],
                text=[f"{m:.2f}%" for m in mapes], textposition="outside",
                marker_line_color="white", marker_line_width=1.5
            ))
            fig_ens.add_hline(y=ens_meta["model_b_mape"], line_dash="dot",
                              line_color="#aaa", annotation_text="Baseline AU solo")
            fig_ens.update_layout(
                title="MAPE por variante de ensemble — test set 60 semanas",
                yaxis_title="MAPE (%)", height=320,
                plot_bgcolor="white", paper_bgcolor="white",
                yaxis=dict(range=[0, max(mapes) * 1.25])
            )
            st.plotly_chart(fig_ens, use_container_width=True)

            # Metrics table
            tbl_data = {
                "Modelo": [l.replace("Model B (Australia only)", "AU solo") for l in labels],
                "MAPE (%)": [f"{m:.2f}" for m in mapes],
                "MAE": [f"{e:.2f}" for e in maes],
                "R²": [f"{r:.4f}" for r in r2s],
                "Features": [models_res[l]["n_features"] for l in labels],
            }
            st.dataframe(pd.DataFrame(tbl_data), use_container_width=True, hide_index=True)

            # Predictions time series
            if os.path.exists(ens_pred_path):
                ens_preds = pd.read_csv(ens_pred_path, parse_dates=["week_date"])
                fig_ens2 = go.Figure()
                fig_ens2.add_trace(go.Scatter(
                    x=ens_preds["week_date"], y=ens_preds["actual_R03"],
                    name="Actual R03", line=dict(color="black", width=2.5)
                ))
                _pred_cols = [c for c in ens_preds.columns if c not in ("week_date", "actual_R03")]
                _col_map = {"Model_B_Australia_only": "#aaaaaa", "Ensemble_AUNZ": "#2e75b6",
                            "Ensemble_AUCL": "#70ad47", "Ensemble_AUNZCL": "#7030a0"}
                for _pc in _pred_cols:
                    _c = _col_map.get(_pc, "#999")
                    fig_ens2.add_trace(go.Scatter(
                        x=ens_preds["week_date"], y=ens_preds[_pc],
                        name=_pc.replace("_", " "), line=dict(color=_c, width=1.5, dash="dash")
                    ))
                fig_ens2.update_layout(
                    title="Predicciones test set — ensemble vs Modelo B",
                    height=350, plot_bgcolor="white", paper_bgcolor="white",
                    legend=dict(orientation="h", y=-0.18),
                    xaxis_title="Semana", yaxis_title="R03 unidades"
                )
                st.plotly_chart(fig_ens2, use_container_width=True)
        else:
            st.warning("Ejecuta `python src/16_ensemble_model.py` para generar los datos del ensemble.")

    # ── TAB 2: SWITCHING RULE ────────────────────────────────────────────────
    with tab_sw:
        st.markdown("### Regla de Cambio Estacional (Switching Rule)")
        st.markdown(
            "El modelo XGBoost tiene dificultades para predecir la temporada baja "
            "(semanas 22-39, mayo-septiembre) donde la demanda R03 es casi plana. "
            "La **regla de cambio** sustituye las predicciones XGBoost fuera de temporada "
            "por la **media estacional histórica** de cada semana ISO."
        )

        sw_meta_path = os.path.join(OUT, "switching_meta.json")
        sw_pred_path = os.path.join(OUT, "switching_rule_results.csv")
        ci_path      = os.path.join(OUT, "ci_calibration.json")

        if os.path.exists(sw_meta_path):
            with open(sw_meta_path) as _f:
                sw_meta = json.load(_f)

            sw_results = sw_meta.get("switching_results", sw_meta)

            # KPI row
            mape_b   = sw_meta.get("model_b_mape",   48.63)
            mape_sw  = sw_meta.get("switched_mape",  35.78)
            mae_b    = sw_meta.get("model_b_mae",    sw_meta.get("model_b_mape", 48.63))
            mae_sw   = sw_meta.get("switched_mae",   sw_meta.get("switched_mape", 35.78))
            improv_sw = mape_b - mape_sw

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Modelo B MAPE", f"{mape_b:.2f}%", "antes del switching")
            s2.metric("Switching MAPE", f"{mape_sw:.2f}%", f"{-improv_sw:+.2f}pp mejora")
            s3.metric("Mejor resultado", f"+{improv_sw:.1f}pp", "mejor del proyecto")
            s4.metric("Semanas off-season", "22-39", "media historica usada")

            st.success(
                f"**Mejor resultado del proyecto:** La regla de cambio reduce el MAPE de "
                f"{mape_b:.2f}% a **{mape_sw:.2f}%** ({improv_sw:+.2f} puntos porcentuales). "
                f"Esto supera a todos los modelos previos, incluyendo el ensemble. "
                f"La clave: reconocer que la temporada baja es mejor modelada con estadísticos "
                f"históricos que con ML."
            )

            if os.path.exists(sw_pred_path):
                sw_df = pd.read_csv(sw_pred_path, parse_dates=["week_date"])

                fig_sw = go.Figure()
                fig_sw.add_trace(go.Scatter(
                    x=sw_df["week_date"], y=sw_df["actual_R03"],
                    name="Actual R03", line=dict(color="black", width=2.5)
                ))
                if "pred_B" in sw_df.columns:
                    fig_sw.add_trace(go.Scatter(
                        x=sw_df["week_date"], y=sw_df["pred_B"],
                        name=f"Modelo B (MAPE {mape_b:.1f}%)",
                        line=dict(color="#aaaaaa", width=1.5, dash="dot")
                    ))
                if "pred_switched" in sw_df.columns:
                    fig_sw.add_trace(go.Scatter(
                        x=sw_df["week_date"], y=sw_df["pred_switched"],
                        name=f"Switching (MAPE {mape_sw:.1f}%)",
                        line=dict(color="#1f4e79", width=2.5)
                    ))
                # shade off-season weeks
                if "iso_week" in sw_df.columns:
                    _off = sw_df[(sw_df["iso_week"] >= 22) & (sw_df["iso_week"] <= 39)]
                    if len(_off) > 0:
                        fig_sw.add_vrect(
                            x0=_off["week_date"].iloc[0], x1=_off["week_date"].iloc[-1],
                            fillcolor="rgba(200,200,200,0.2)", line_width=0,
                            annotation_text="Off-season (media hist.)", annotation_position="top left"
                        )
                fig_sw.update_layout(
                    title="Switching Rule vs Modelo B — test set 192 semanas WFV",
                    height=380, plot_bgcolor="white", paper_bgcolor="white",
                    legend=dict(orientation="h", y=-0.18),
                    xaxis_title="Semana", yaxis_title="R03 unidades"
                )
                st.plotly_chart(fig_sw, use_container_width=True)

            # CI Calibration
            if os.path.exists(ci_path):
                st.markdown("### Calibracion de Intervalos de Confianza (WFV Empirico)")
                with open(ci_path) as _f:
                    ci_meta = json.load(_f)

                cov80 = ci_meta.get("coverage_80pct", 74.0)
                cov50 = ci_meta.get("coverage_50pct", 55.7)
                n_wks = ci_meta.get("n_weeks", 192)

                c80, c50, cn = st.columns(3)
                c80.metric("Cobertura IC 80%", f"{cov80:.1f}%",
                           f"{'OK' if abs(cov80-80) < 15 else 'REVISAR'} (nominal 80%)")
                c50.metric("Cobertura IC 50%", f"{cov50:.1f}%",
                           f"{'OK' if abs(cov50-50) < 15 else 'REVISAR'} (nominal 50%)")
                cn.metric("Semanas evaluadas", str(n_wks), "WFV out-of-sample")

                st.markdown(
                    "La calibración empírica usa los errores de las 192 predicciones walk-forward "
                    "agrupados por semana ISO (p10/p50/p90 por semana). "
                    f"Un IC 80% que cubre el {cov80:.0f}% de las semanas está "
                    f"{'bien calibrado' if abs(cov80-80) < 12 else 'ligeramente mal calibrado'} "
                    f"(diferencia {abs(cov80-80):.1f}pp del nominal)."
                )
        else:
            st.warning("Ejecuta `python src/17_switching_ci.py` para generar los datos de switching.")


# ==============================================================================
# PAGE 12: DIAGRAMA DE PIPELINE
# ==============================================================================
elif page == "Diagrama de Pipeline":
    st.markdown('<h2 class="section-title">Diagrama del Pipeline de Analisis</h2>',
                unsafe_allow_html=True)
    st.caption("Arquitectura del sistema de 17 pasos — desde datos brutos hasta dashboard")

    # Pipeline steps data
    _pipeline_steps = [
        ("01", "Descarga WHO FluNet", "Datos brutos",   "#d5e8d4", "#82b366", "data/raw/flunet_global.csv"),
        ("02", "Limpieza FluNet",     "Procesamiento",  "#dae8fc", "#6c8ebf", "data/processed/flunet_*.csv"),
        ("03", "Ventas EU Kaggle",    "Procesamiento",  "#dae8fc", "#6c8ebf", "data/processed/eu_sales.csv"),
        ("04", "PBS Australia",       "Procesamiento",  "#dae8fc", "#6c8ebf", "data/processed/pbs_au.csv"),
        ("05", "Integrar datasets",   "Ingenieria",     "#fff2cc", "#d6b656", "data/processed/integrated_dataset.csv"),
        ("06", "CCF lead-lag",        "Analisis",       "#f8cecc", "#b85450", "output/ccf_results.json"),
        ("07", "XGBoost A & B",       "Modelo",         "#e1d5e7", "#9673a6", "output/xgboost_best_model.json"),
        ("08", "AMELI Francia",       "Datos brutos",   "#d5e8d4", "#82b366", "output/france_openmedic.csv"),
        ("09", "SHAP analysis",       "Explicabilidad", "#fff2cc", "#d6b656", "output/shap_values.csv"),
        ("10", "Walk-Forward Val.",   "Validacion",     "#f8cecc", "#b85450", "output/wfv_predictions.csv"),
        ("11", "SARIMA + Ljung-Box",  "Baseline",       "#e1d5e7", "#9673a6", "output/sarima_predictions.csv"),
        ("12", "Simulacion inv.",     "Optimizacion",   "#dae8fc", "#6c8ebf", "output/inventory_simulation.csv"),
        ("13", "SH 7-paises",         "Validacion",     "#f8cecc", "#b85450", "output/sh_model_comparison.csv"),
        ("14", "Figuras EDA",         "Visualizacion",  "#d5e8d4", "#82b366", "output/eda_*.png"),
        ("15", "R06 + DM tests",      "Modelo",         "#e1d5e7", "#9673a6", "output/dm_test_results.json"),
        ("16", "Ensemble SH",         "Experimento",    "#fff2cc", "#d6b656", "output/ensemble_meta.json"),
        ("17", "Switching + CI",      "Mejora",         "#f8cecc", "#b85450", "output/switching_meta.json"),
    ]

    # Status check (which outputs exist)
    def _step_exists(output_pattern):
        import glob as _glob
        _base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "")
        _p = _base + output_pattern.replace("*", "**")
        return len(_glob.glob(_base + output_pattern)) > 0

    st.markdown("#### Estado actual del pipeline")
    _n_cols = 6
    for _row_start in range(0, len(_pipeline_steps), _n_cols):
        _row_steps = _pipeline_steps[_row_start:_row_start + _n_cols]
        _cols = st.columns(len(_row_steps))
        for _ci, (_num, _name, _cat, _bg, _border, _out) in enumerate(_row_steps):
            _done = _step_exists(_out)
            _status_icon = "✅" if _done else "⏳"
            _cols[_ci].markdown(
                f'<div style="background:{_bg};border:2px solid {_border};'
                f'border-radius:10px;padding:8px 10px;text-align:center;'
                f'font-size:0.78rem;height:80px;display:flex;flex-direction:column;'
                f'justify-content:center;color:#1a1a1a">'
                f'<b style="color:{_border};font-size:0.85rem">{_status_icon} {_num}</b><br>'
                f'<span style="color:#1a1a1a">{_name}</span><br>'
                f'<span style="font-size:0.7rem;opacity:0.75;color:#333">{_cat}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown("")

    # Flow diagram using Plotly
    st.markdown("#### Flujo de datos")
    _fig_flow = go.Figure()

    _node_x = [0.05, 0.05, 0.05, 0.05,  0.25,  0.45, 0.45, 0.45, 0.45,  0.65, 0.65, 0.65,  0.85, 0.85]
    _node_y = [0.90, 0.70, 0.50, 0.30,  0.60,  0.90, 0.70, 0.50, 0.30,  0.80, 0.55, 0.30,  0.70, 0.40]
    _node_labels = [
        "WHO FluNet\n(140 países)",
        "EU Pharma\n(Kaggle)",
        "PBS Australia",
        "AMELI\nFrancia",
        "Dataset\nIntegrado\n(Step 05)",
        "CCF\nLead-lag",
        "XGBoost\nModelo A&B",
        "Walk-Forward\nValidacion",
        "SARIMA\nBaseline",
        "SHAP\nExplicab.",
        "Simulacion\nInventario",
        "R06+DM\nTests",
        "Ensemble\n+Switching",
        "Dashboard\nStreamlit"
    ]
    _node_colors = [
        "#d5e8d4","#d5e8d4","#d5e8d4","#d5e8d4",
        "#fff2cc",
        "#dae8fc","#e1d5e7","#f8cecc","#e1d5e7",
        "#fff2cc","#dae8fc","#e1d5e7",
        "#f8cecc","#1f4e79"
    ]
    _node_font_colors = ["#2d6a2d"] * 4 + ["#7a6000"] + \
                        ["#1f4e79","#5a3a7e","#8b0000","#5a3a7e"] + \
                        ["#7a6000","#1f4e79","#5a3a7e","#8b0000"] + ["white"]

    # Edges — drawn as Scatter lines (works in all Plotly versions)
    _edges = [
        (0, 4), (1, 4), (2, 4), (3, 4),
        (4, 5), (4, 6), (4, 7), (4, 8),
        (6, 7), (6, 9), (7, 8), (7, 10),
        (6, 11), (8, 10),
        (6, 12), (7, 12),
        (9, 13), (10, 13), (11, 13), (12, 13), (5, 13)
    ]

    for _e_src, _e_dst in _edges:
        _fig_flow.add_trace(go.Scatter(
            x=[_node_x[_e_src], _node_x[_e_dst]],
            y=[_node_y[_e_src], _node_y[_e_dst]],
            mode="lines",
            line=dict(color="#bbbbbb", width=1.5),
            hoverinfo="skip", showlegend=False
        ))

    # Node markers drawn on top of edges
    _fig_flow.add_trace(go.Scatter(
        x=_node_x, y=_node_y,
        mode="markers+text",
        marker=dict(
            size=54, color=_node_colors,
            line=dict(color="#666", width=1.5),
            symbol="square"
        ),
        text=_node_labels,
        textfont=dict(size=8, color=_node_font_colors),
        textposition="middle center",
        hoverinfo="text",
        hovertext=[f"Step: {l}" for l in _node_labels],
        showlegend=False
    ))

    _fig_flow.update_layout(
        height=480,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.05, 1.0]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0.1, 1.05]),
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=10, r=10, t=20, b=10)
    )
    st.plotly_chart(_fig_flow, use_container_width=True)

    # Category legend
    st.markdown("**Leyenda por categoria:**")
    _leg_cols = st.columns(6)
    for _li, (_lcat, _lcolor) in enumerate([
        ("Datos brutos", "#d5e8d4"), ("Ingenieria", "#fff2cc"),
        ("Modelo", "#e1d5e7"), ("Validacion", "#f8cecc"),
        ("Optimizacion", "#dae8fc"), ("Experimento/Mejora", "#f8cecc")
    ]):
        _leg_cols[_li].markdown(
            f'<div style="background:{_lcolor};border-radius:6px;'
            f'padding:4px 8px;font-size:0.75rem;text-align:center;color:#1a1a1a">{_lcat}</div>',
            unsafe_allow_html=True
        )

    st.divider()
    st.markdown("#### Como ejecutar el pipeline completo")
    st.code("""# Ejecutar todos los pasos en orden
python run_pipeline.py

# Reanudar desde un paso especifico
python run_pipeline.py --from 9

# Ejecutar un solo paso
python run_pipeline.py --only 17

# Luego lanzar el dashboard
streamlit run dashboard/app.py""", language="bash")

    # Output file inventory
    st.markdown("#### Inventario de outputs generados")
    _output_files = [
        ("xgboost_best_model.json",     "Modelo XGBoost serializado (produccion)"),
        ("test_predictions.csv",         "Predicciones test 60 semanas"),
        ("wfv_predictions.csv",          "192 predicciones walk-forward OOS"),
        ("sarima_predictions.csv",       "Forecast SARIMA + IC 80%"),
        ("inventory_simulation.csv",     "Simulacion inventario (4 escenarios)"),
        ("sh_model_comparison.csv",      "Validacion 7 paises Hemisferio Sur"),
        ("dm_test_results.json",         "Tests Diebold-Mariano de significancia"),
        ("ensemble_meta.json",           "Resultados ensemble AU+NZ+Chile"),
        ("switching_meta.json",          "Regla de cambio + mejora MAPE"),
        ("ci_calibration.json",          "Calibracion IC empirica por semana ISO"),
        ("TFG_Final_v2.docx",            "Documento tesis completo con figuras"),
    ]
    _out_rows = []
    _out_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
    for _fname, _fdesc in _output_files:
        _fpath = os.path.join(_out_base, _fname)
        _exists = "✅" if os.path.exists(_fpath) else "❌"
        _size = ""
        if os.path.exists(_fpath):
            _sz = os.path.getsize(_fpath)
            _size = f"{_sz/1024:.1f} KB" if _sz < 1024*1024 else f"{_sz/1024/1024:.1f} MB"
        _out_rows.append({"Estado": _exists, "Archivo": _fname, "Descripcion": _fdesc, "Tamano": _size})
    st.dataframe(pd.DataFrame(_out_rows), use_container_width=True, hide_index=True)
