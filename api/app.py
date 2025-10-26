# app.py
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import pandas as pd
import numpy as np
import joblib

from utils_viz import (
    fig_to_html, plot_feature_importance, plot_confusion,
    plot_roc_ovr, plot_pred_proba_hist
)

app = FastAPI(title="Random Forest App")

# Ajusta estas rutas/nombres:
CSV_PATH = "/content/drive/MyDrive/Dataset/datasets/TotalFeatures-ISCXFlowMeter.csv"
MODEL_PATH = "/content/model_store/rf_model.joblib"  # o donde lo guardes
TARGET = "duration"  # o tu columna objetivo REAL si quieres evaluar

# Carga modelo
model = joblib.load(MODEL_PATH)

# Si guardaste la lista de columnas usadas por el modelo:
try:
    feature_names = joblib.load(MODEL_PATH.replace(".joblib", "_features.joblib"))
except:
    feature_names = None

@app.get("/api/predict/csv", response_class=HTMLResponse)
def predict_csv():
    df = pd.read_csv(CSV_PATH)
    # Seleccionar features
    X = df[feature_names] if feature_names else df.select_dtypes(include=[np.number]).copy()

    # --- Predicciones ---
    y_pred = model.predict(X)
    try:
        proba = model.predict_proba(X)
    except Exception:
        # Algunos modelos no tienen predict_proba
        proba = np.expand_dims(model.decision_function(X), 1)

    # --- Construcción de gráficas ---
    figs_html = []

    # 1) Importancia de features (top 20)
    figs_html.append(fig_to_html(plot_feature_importance(model, X.columns, top=20)))

    # 2) Histograma de probabilidades (binaria) o score
    if proba.ndim == 2:
        figs_html.append(fig_to_html(plot_pred_proba_hist(proba)))

    # 3) Matriz de confusión (si tienes y reales en el CSV)
    if TARGET in df.columns:
        y_true = df[TARGET].values
        # asegurar mismo tipo
        try:
            figs_html.append(fig_to_html(plot_confusion(y_true, y_pred)))
        except Exception:
            pass

        # 4) Curva ROC (binaria o multiclase OVR)
        try:
            classes = np.unique(y_true)
            figs_html.append(fig_to_html(plot_roc_ovr(y_true, proba, classes)))
        except Exception:
            pass

    # --- Tabla con primeras filas + predicción ---
    out = df.copy()
    out["__pred__"] = y_pred
    if proba.ndim == 2 and proba.shape[1] >= 2:
        out["__proba__"] = proba.max(1)
    table_html = out.head(100).to_html(index=False)

    # --- Página HTML simple ---
    html = f"""
    <html>
    <head>
        <title>Random Forest App</title>
        <meta charset="utf-8"/>
        <style>
            body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; }}
            h1 {{ margin-bottom: 8px; }}
            .grid {{ display: grid; grid-template-columns: 1fr; gap: 28px; }}
            @media(min-width: 1200px) {{ .grid {{ grid-template-columns: 1fr 1fr; }} }}
            .card {{ background: #fff; border-radius: 12px; padding: 16px; box-shadow: 0 1px 6px rgba(0,0,0,.06); }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ border-bottom: 1px solid #eee; padding: 8px; font-size: 14px; }}
            thead th {{ position: sticky; top: 0; background: #fafafa; }}
        </style>
    </head>
    <body>
        <h1>Random Forest App</h1>
        <p>Se muestran las primeras 100 filas con su predicción y gráficas de diagnóstico.</p>
        <div class="card">{table_html}</div>
        <div class="grid">
            {"".join(f'<div class="card">{f}</div>' for f in figs_html)}
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html)
from sklearn import tree
import matplotlib.pyplot as plt
import io, base64

def tree_png_html(rf, feature_names, class_names=None, tree_index=0, max_depth=3):
    estimator = rf.estimators_[tree_index]
    fig, ax = plt.subplots(figsize=(14, 8))
    tree.plot_tree(estimator, feature_names=feature_names, class_names=class_names,
                   filled=True, max_depth=max_depth, fontsize=8)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode()
    return f'<img src="data:image/png;base64,{data}" />'
