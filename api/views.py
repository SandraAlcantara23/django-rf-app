# api/views.py
import os
from pathlib import Path

import numpy as np
import pandas as pd
from django.http import HttpResponseBadRequest
from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .ml import MODEL_PATH, load_meta, model_exists, load_model, predict_df
from .utils_viz import (
    fig_to_html,
    plot_confusion,
    plot_feature_importance,
    plot_pred_proba_hist,
    plot_roc_ovr,
    tree_png_html,
)


# -------- util --------
def _as_bool(v) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y", "t"}


# Activa el render del árbol solo si quieres (o ponlo en Render env vars)
ENABLE_TREE_IMAGE = _as_bool(os.getenv("ENABLE_TREE_IMAGE", "0"))


# -------- endpoints --------
@api_view(["GET"])
def health(request):
    return Response({"status": "ok"})


def home(request):
    """
    Página inicial: muestra estado del modelo y columnas esperadas (metadatos).
    No deserializa el modelo para que cargue muy rápido.
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
    POST con un CSV -> devuelve tabla + gráficas (Plotly) y, opcionalmente, un árbol PNG.
    """
    if request.method != "POST":
        return HttpResponseBadRequest("Método no permitido")

    if not model_exists():
        return HttpResponseBadRequest("No hay modelo entrenado aún")

    # Admite 'csv_file' o 'file'
    f = request.FILES.get("csv_file") or request.FILES.get("file")
    if not f:
        return HttpResponseBadRequest("Falta archivo CSV")

    # Leer CSV
    try:
        df = pd.read_csv(f)
    except Exception as e:
        return HttpResponseBadRequest(f"Error leyendo CSV: {e}")

    try:
        # Carga modelo + metadatos
        model, meta = load_model()
        features = meta.get("features")

        # Selección de X coherente con entrenamiento
        if features:
            cols = [c for c in features if c in df.columns]
            if not cols:
                return HttpResponseBadRequest(
                    "El CSV no contiene columnas de entrenamiento."
                )
            X = df[cols].copy()
        else:
            # Fallback: numéricas
            X = df.select_dtypes(include=[np.number]).copy()
            if X.empty:
                return HttpResponseBadRequest(
                    "No se encontraron columnas numéricas para predecir."
                )

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
                proba = np.atleast_2d(y_pred).T  # fallback mínimo

        # --------- Gráficas ----------
        figs_html = []

        # Importancias
        try:
            figs_html.append(fig_to_html(plot_feature_importance(model, X.columns, top=20)))
        except Exception:
            pass

        # Histograma de probas/scores
        if proba is not None and getattr(proba, "ndim", 0) >= 1:
            try:
                figs_html.append(fig_to_html(plot_pred_proba_hist(proba)))
            except Exception:
                pass

        # Confusión + ROC si el CSV trae la columna objetivo
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

        # Árbol del bosque (opcional)
        if ENABLE_TREE_IMAGE and hasattr(model, "estimators_") and len(model.estimators_) > 0:
            figs_html.append(
                tree_png_html(
                    model,
                    X.columns,
                    class_names=None,  # la función deduce si es clasificación
                    tree_index=0,
                    max_depth=3,
                )
            )

        # Tabla de salida
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


@api_view(["POST"])
def api_predict(request):
    """
    JSON:
      {"rows":[{"feat1":1,"feat2":2.3,...}, ...]}
    Respuesta:
      {"predictions":[...]}
    """
    if not model_exists():
        return Response({"error": "No model"}, status=400)

    rows = request.data.get("rows")
    if not isinstance(rows, list):
        return Response(
            {"error": "Formato inválido: se espera 'rows' como lista de objetos"},
            status=400,
        )

    try:
        df = pd.DataFrame(rows)
        _, preds = predict_df(df)
        return Response({"predictions": preds})
    except Exception as e:
        return Response({"error": str(e)}, status=400)
