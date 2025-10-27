# api/ml.py
from pathlib import Path
from typing import List, Tuple
import joblib
import pandas as pd
import warnings
from sklearn.base import InconsistentVersionWarning

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "model_store"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "rf_model.joblib"
META_PATH = MODEL_DIR / "rf_meta.joblib"   # {'features': [...], 'target': '...'}

def model_exists() -> bool:
    return MODEL_PATH.exists() and META_PATH.exists()

def load_model() -> Tuple[object, dict]:
    """Carga el modelo (sklearn) y los metadatos, ignorando el warning de versión."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
        model = joblib.load(MODEL_PATH)
    meta = joblib.load(META_PATH)
    return model, meta

def load_meta() -> dict:
    """Carga SOLO los metadatos (rápido, sin deserializar el modelo)."""
    if not META_PATH.exists():
        return {}
    return joblib.load(META_PATH)

def save_model(model, features: List[str], target: str):
    joblib.dump(model, MODEL_PATH)
    joblib.dump({"features": features, "target": target}, META_PATH)

def ensure_columns(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en CSV: {missing}")
    return df[features].copy()

def predict_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    model, meta = load_model()
    features = meta.get("features", [])
    X = ensure_columns(df, features) if features else df.select_dtypes(include="number").copy()
    preds = model.predict(X)
    out = df.copy()
    out["prediction"] = preds
    return out, preds.tolist()
