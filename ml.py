# api/ml.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict, Any
import warnings

from sklearn.base import InconsistentVersionWarning
import joblib
import pandas as pd


# --- Rutas del proyecto/modelo ---
BASE_DIR = Path(__file__).resolve().parent.parent  # .../src
MODEL_DIR = BASE_DIR / "model_store"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH: Path = MODEL_DIR / "rf_model.joblib"
META_PATH: Path = MODEL_DIR / "rf_meta.joblib"   # {'features': [...], 'target': '...'}


def model_exists() -> bool:
    """Devuelve True si existen ambos archivos del modelo."""
    return MODEL_PATH.exists() and META_PATH.exists()


def load_model() -> Tuple[Any, Dict[str, Any]]:
    """
    Carga el modelo y sus metadatos.
    Silencia InconsistentVersionWarning (sklearn 1.6 vs 1.7, etc).
    """
    if not model_exists():
        raise FileNotFoundError(
            "No se encontró el modelo entrenado (model_store/rf_model.joblib o rf_meta.joblib)."
        )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
        model = joblib.load(MODEL_PATH)
        meta: Dict[str, Any] = joblib.load(META_PATH)

    if "features" not in meta:
        raise KeyError("El archivo rf_meta.joblib no contiene la clave 'features'.")

    return model, meta


def save_model(model: Any, features: List[str], target: str) -> None:
    """Guarda el estimador y los metadatos (lista de columnas y nombre del target)."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump({"features": list(features), "target": target}, META_PATH)


def ensure_columns(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Verifica que el CSV traiga todas las columnas esperadas y las reordena exactamente
    como en entrenamiento.
    """
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {missing}")
    return df.loc[:, features].copy()


def predict_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, List]:
    """
    Aplica el modelo sobre el DataFrame `df` (debe contener las columnas de `features`)
    y devuelve (dataframe_con_predicción, lista_de_predicciones).
    """
    model, meta = load_model()
    X = ensure_columns(df, list(meta["features"]))
    preds = model.predict(X)

    out = df.copy()
    out["prediction"] = preds

    # Aseguramos lista Python (para JSON en API)
    try:
        pred_list = preds.tolist()
    except AttributeError:
        pred_list = list(preds)

    return out, pred_list
