# api/views.py
import os
from pathlib import Path
from django.shortcuts import render
from django.http import HttpResponseBadRequest
from rest_framework.decorators import api_view
from rest_framework.response import Response
import pandas as pd
import numpy as np

from .ml import model_exists, load_model, load_meta, predict_df, MODEL_PATH

from .utils_viz import (
    fig_to_html, plot_feature_importance, plot_confusion,
    plot_roc_ovr, plot_pred_proba_hist, tree_png_html
)

ENABLE_TREE_IMAGE = os.getenv("ENABLE_TREE_IMAGE", "0") == "1"

@api_view(["GET"])
def health(request):
    return Response({"status": "ok"})

def home(request):
    """
    Página inicial: no deserializamos el modelo (rápido). Solo leemos meta.
    """
    ready = model_exists()
    feature_list = []
    model_filename = ""

    if ready:
        meta = load_meta() or {}
        feature_list = meta.get("features", [])
        model_filename = Path(str(MODEL_PATH)).name

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
    ready = model_exists()
    ctx = {"model_ready": ready, "model_path": Path(str(MODEL_PATH)).name if ready else ""}
    return render(request, "base.html", ctx)

def predict_csv(request):
    """
    Recibe CSV por POST y devuelve:
      - Tabla con predicciones (primeras 50 filas)
      - Gráficas (feature importance, hist proba, confusión y ROC si hay y_true)
      - Árbol del RandomForest (opcional: ENABLE_TREE_IMAGE=1)
    """
    if request.method != "POST":
        return HttpResponseBadRequest("Método no permitido")

    if not model_exists():
        return HttpResponseBadRequest("No hay modelo entrenado aún")

    f = request.FILES.get("csv_file") or request.FILES.get("file")
    if not f:
        return HttpResponseBadRequest("Falta archivo CSV")

    try:
        df = pd.read_csv(f)
    except Exception as e:
        return HttpResponseBadRequest(f"Error leyendo CSV: {e}")

    try:
        model, meta = load_model()
        features = meta.get("features")
        # Selección de X
        if features:
            cols = [c for c in features if c in df.columns]
            if not cols:
                return HttpResponseBadRequest("El CSV no contiene columnas de entrenamiento.")
            X = df[cols].copy()
        else:
            X = df.select_dtypes(include=[np.number]).copy()

        # Predicción
        y_pred = model.predict(X)

        # Probabilidades / scores
        try:
            proba = model.predict_proba(X)
        except Exception:
            try:
                scores = model.decision_function(X)
                proba = np.atleast_2d(scores).T if getattr(scores, "ndim", 1) == 1 else scores
            except Exception:
                proba = np.atleast_2d(y_pred).T

        figs_html = []
        # Importancias
        try:
            figs_html.append(fig_to_html(plot_feature_importance(model, X.columns, top=20)))
        except Exception:
            pass
        # Hist proba
        if proba is not None and getattr(proba, "ndim", 0) >= 1:
            try:
                figs_html.append(fig_to_html(plot_pred_proba_hist(proba)))
            except Exception:
                pass
        # Confusión + ROC si hay target
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

        # Árbol (opcional para evitar coste de fuentes en Render)
        if ENABLE_Boolean(ENABLE_TREE_IMAGE) and hasattr(model, "estimators_") and len(getattr(model, "estimators_", [])) > 0:
            try:
                figs_html.append(
                    tree_png_html(model, X.columns, class_names=None, tree_index=0, max_depth=3)
                )
            except Exception:
                # Si falla por fuentes, simplemente omitimos
                pass

        out = df.copy()
        out["prediction"] = y_pred
        if getattr(proba, "ndim", 0) == 2 and proba.shape[1] >= 2:
            out["pred_proba"] = proba.max(axis=1)

        context = {
            "table_html": out.head(50).to_html(index=False),
            "figs_html": figs_html,
        }
        return render(request, "predict_result.html", context)

    except Exception as e:
        return HttpResponseBadRequest(f"Error durante la predicción: {e}")

def ENABLE_Boolean(v):
    return str(v).strip().lower() in {"1", "true", "yes", "y", "t"}

@api_view(["POST"])
def api_predict(request):
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
