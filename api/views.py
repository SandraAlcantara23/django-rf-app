# api/views.py
from pathlib import Path
from django.shortcuts import render
from django.http import HttpResponseBadRequest
from rest_framework.decorators import api_view
from rest_framework.response import Response
import pandas as pd
import numpy as np

# Tu módulo de ML
from .ml import model_exists, load_model, predict_df, MODEL_PATH

# ⛔️ Importante: NO importamos utils_viz arriba para evitar cargar matplotlib/plotly
# al entrar a la home. Lo haremos dentro de la vista predict_csv.

# === Endpoints simples ===
@api_view(["GET"])
def health(request):
    return Response({"status": "ok"})

def home(request):
    """
    Página inicial: estado del modelo y features.
    """
    ready = model_exists()
    feature_list = []
    model_filename = ""

    if ready:
        model, meta = load_model()
        feature_list = meta.get("features", [])
        model_filename = Path(str(MODEL_PATH)).name  # evita .name sobre str

    return render(
        request,
        "home.html",
        {
            "model_ready": ready,
            "model_path": model_filename,
            "feature_list": ", ".join(feature_list),
        },
    )

def train_page(request):
    """Página simple de estado/entrenamiento."""
    ready = model_exists()
    ctx = {"model_ready": ready, "model_path": Path(str(MODEL_PATH)).name if ready else ""}
    return render(request, "base.html", ctx)

# === Predicción desde CSV con tabla + gráficas ===
def predict_csv(request):
    """
    Recibe CSV por POST y devuelve:
      - Tabla con predicciones (primeras 50 filas)
      - Gráficas (feature importance, hist proba, confusión y ROC si hay y_true)
      - Un árbol del RandomForest (tree_index=0, max_depth=3)
    """
    if request.method != "POST":
        return HttpResponseBadRequest("Método no permitido")

    if not model_exists():
        return HttpResponseBadRequest("No hay modelo entrenado aún")

    # Acepta ambos nombres de campo
    f = request.FILES.get("csv_file") or request.FILES.get("file")
    if not f:
        return HttpResponseBadRequest("Falta archivo CSV")

    # Leer CSV
    try:
        df = pd.read_csv(f)
    except Exception as e:
        return HttpResponseBadRequest(f"Error leyendo CSV: {e}")

    try:
        # Modelo + metadatos
        model, meta = load_model()
        features = meta.get("features")  # puede ser None

        # Selección de X coherente con el entrenamiento
        if features:
            cols = [c for c in features if c in df.columns]
            if not cols:
                return HttpResponseBadRequest(
                    "El CSV no contiene ninguna de las columnas de entrenamiento."
                )
            X = df[cols].copy()
        else:
            X = df.select_dtypes(include=[np.number]).copy()

        # Predicciones
        y_pred = model.predict(X)

        # Probabilidades / scores
        try:
            proba = model.predict_proba(X)
        except Exception:
            try:
                scores = model.decision_function(X)
                proba = np.atleast_2d(scores).T if scores.ndim == 1 else scores
            except Exception:
                proba = np.atleast_2d(y_pred).T  # fallback mínimo

        # --- Gráficas ---
        # 👇 Importamos utils_viz SOLO aquí (cuando realmente se piden gráficas)
        from .utils_viz import (
            fig_to_html, plot_feature_importance, plot_confusion,
            plot_roc_ovr, plot_pred_proba_hist, tree_png_html
        )

        figs_html = []
        # 1) Importancia de features
        figs_html.append(fig_to_html(plot_feature_importance(model, X.columns, top=20)))

        # 2) Histograma de probabilidades
        if proba is not None and proba.ndim >= 1:
            figs_html.append(fig_to_html(plot_pred_proba_hist(proba)))

        # 3) Matriz de confusión y 4) ROC si existe la columna objetivo
        TARGET = meta.get("target") or "duration"
        if TARGET in df.columns:
            y_true = df[TARGET].values
            try:
                figs_html.append(fig_to_html(plot_confusion(y_true, y_pred)))
            except Exception:
                pass
            try:
                classes = np.unique(y_true)
                figs_html.append(fig_to_html(plot_roc_ovr(y_true, proba, classes)))
            except Exception:
                pass

        # 5) Árbol del bosque (índice 0, profundidad 3)
        try:
            figs_html.append(
                tree_png_html(model, X.columns, class_names=None, tree_index=0, max_depth=3)
            )
        except Exception:
            pass

        # --- Tabla de salida ---
        out = df.copy()
        out["prediction"] = y_pred
        if proba.ndim == 2 and proba.shape[1] >= 2:
            out["pred_proba"] = proba.max(axis=1)

        context = {
            "table_html": out.head(50).to_html(index=False),
            "figs_html": figs_html,
        }
        return render(request, "predict_result.html", context)

    except Exception as e:
        return HttpResponseBadRequest(f"Error durante la predicción: {e}")

# === API JSON para predicción por filas ===
@api_view(["POST"])
def api_predict(request):
    """
    Recibe JSON:
    {
      "rows": [
        {"feat1": 1, "feat2": 3.2, ...},
        ...
      ]
    }
    Devuelve: {"predictions": [...]}
    """
    if not model_exists():
        return Response({"error": "No model"}, status=400)

    rows = request.data.get("rows")
    if not isinstance(rows, list):
        return Response({"error": "Formato inválido, se espera 'rows' lista de objetos"}, status=400)

    try:
        df = pd.DataFrame(rows)
        _, preds = predict_df(df)
        return Response({"predictions": preds})
    except Exception as e:
        return Response({"error": str(e)}, status=400)
