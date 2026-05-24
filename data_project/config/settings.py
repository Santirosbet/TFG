"""
settings.py — Configuración centralizada del proyecto.
NUNCA hardcodear claves aquí. Todas las claves van en .env (no en git).
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# === RUTAS BASE ===
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
OUTPUT_DIR = BASE_DIR / "output"
MODELS_DIR = BASE_DIR / "models"
SRC_DIR = BASE_DIR / "src"

# === APIs — SIEMPRE desde variables de entorno ===
NOTION_TOKEN = os.getenv("NOTION_TOKEN", "")
OPENMETEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
GOOGLE_TRENDS_DELAY = float(os.getenv("GOOGLE_TRENDS_DELAY", "2.0"))

# === PARÁMETROS DEL MODELO ===
RANDOM_SEED = 42
SWITCHING_RULE_SUMMER_WEEKS = list(range(22, 40))  # Semanas 22-39
LAG_AU_FLU = 26          # CCF peak = 28w, mejor performance práctico = 26w
WFV_FOLDS = 48
TARGET_CI_COVERAGE = 0.80

# === NOTION PAGE IDs ===
NOTION_PAGES = {
    "hub":         "35c4a293c2b58166b26fda2d479cc133",
    "contexto":    "35c4a293c2b58144b8f6d1598ca43298",
    "modelos":     "35c4a293c2b581f4800cf3abc141d9fb",
    "datos":       "35c4a293c2b581c6a0efe04c58447665",
    "numeros":     "35c4a293c2b58124b857f8878e5bda93",
}

# === DATASETS ===
MAIN_DATASET = PROCESSED_DIR / "integrated_dataset.csv"
ENHANCED_DATASET = PROCESSED_DIR / "integrated_dataset_v2.csv"

# === OUTPUTS ===
MODEL_META = OUTPUT_DIR / "model_meta.json"
PROPHET_META = OUTPUT_DIR / "prophet_meta.json"
SWITCHING_META = OUTPUT_DIR / "switching_meta.json"
ENSEMBLE_META = OUTPUT_DIR / "ensemble_meta.json"
WFV_META = OUTPUT_DIR / "wfv_meta.json"
CONFORMAL_META = OUTPUT_DIR / "conformal_meta.json"
LGBM_META = OUTPUT_DIR / "lgbm_meta.json"
STACKING_META = OUTPUT_DIR / "stacking_meta.json"

# === CIUDADES PARA DATOS METEOROLÓGICOS ===
WEATHER_CITIES = [
    {"name": "Madrid",    "lat": 40.4168, "lon": -3.7038,  "weight": 0.10},
    {"name": "Paris",     "lat": 48.8566, "lon":  2.3522,  "weight": 0.15},
    {"name": "Berlin",    "lat": 52.5200, "lon": 13.4050,  "weight": 0.12},
    {"name": "Rome",      "lat": 41.9028, "lon": 12.4964,  "weight": 0.10},
    {"name": "Amsterdam", "lat": 52.3676, "lon":  4.9041,  "weight": 0.08},
    {"name": "Warsaw",    "lat": 52.2297, "lon": 21.0122,  "weight": 0.10},
    {"name": "Bucharest", "lat": 44.4268, "lon": 26.1025,  "weight": 0.09},
]


def validate_env():
    """Valida que las variables de entorno críticas están definidas."""
    missing = []
    if not NOTION_TOKEN:
        missing.append("NOTION_TOKEN")
    if missing:
        print(f"[settings] Variables de entorno no definidas: {missing}")
        print("  Crea un archivo .env en la raíz del proyecto con estas claves.")
        print("  Ver .env.example como plantilla.")
    return len(missing) == 0


def ensure_dirs():
    """Crea los directorios necesarios si no existen."""
    for d in [RAW_DIR, PROCESSED_DIR, EXTERNAL_DIR, OUTPUT_DIR, MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
