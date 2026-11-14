from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
SQL_DIR = ROOT / "sql"
DOCS_DIR = ROOT / "docs"

DB_PATH = DATA_DIR / "wells.db"
LONG_PARQUET = DATA_DIR / "long.parquet"
WIDE_PARQUET = DATA_DIR / "wide.parquet"

HF_BASE_URL = (
    "https://huggingface.co/datasets/"
    "Arailym-tleubayeva/NK-Oil-Well-Sensor-Monitoring-v2/resolve/main"
)
SPLITS = ("train", "test")

FOUNTAIN_PARAMS = ["tubing_pressure", "casing_pressure", "line_pressure"]
PUMP_PARAMS = ["SPM", "pump_fillage", "min_rod_weight", "max_rod_weight", "dynamometer_area"]
PARAMETERS = FOUNTAIN_PARAMS + PUMP_PARAMS
WELL_TYPES = ("fountain", "sucker_rod_pump")

AREA_SHIFT_MONTH = "2026-05"
NK7_REGIME_START = "2026-06-01"
RANDOM_STATE = 42


def ensure_dirs():
    for path in (RAW_DIR, FIGURES_DIR, SQL_DIR, DOCS_DIR):
        path.mkdir(parents=True, exist_ok=True)
