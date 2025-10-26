# api/utils_viz.py
import os
import io
import base64
import numpy as np

# Plotly (gráficas interactivas, ligeras para servidor)
import plotly.express as px
import plotly.graph_objects as go

# Métricas
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize


def _mpl_setup():
    """
    Configura Matplotlib para servidor:
    - backend 'Agg' (sin ventana)
    - cache de fuentes en /tmp (escritura permitida en Render/Heroku)
    Se importa perezosamente para evitar coste en la home.
    """
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache")
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    return plt


def fig_to_html(fig):
    """Devuelve el HTML embebible de la figura Plotly (incluye plotly.js desde CDN)."""
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def plot_feature_importance(model, feature_names, top=20):
    """Barra horizontal con las importancias del modelo (si existen)."""
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        importances = np.zeros(len(feature_names))
    order = np.argsort(importances)[::-1][:top]
    x_vals = importances[order][::-1]
    y_vals = np.array(feature_names)[order][::-1]
    fig = px.bar(
        x=x_vals,
        y=y_vals,
        orientation="h",
        title=f"Importancia de características (top {min(top, len(feature_names))})",
        labels={"x": "Importancia", "y": "Feature"},
    )
    fig.update_layout(height=600, margin=dict(l=80, r=20, t=60, b=40))
    return fig


def plot_confusion(y_true, y_pred):
    """Matriz de confusión."""
    labels = np.unique(np.concatenate([np.unique(y_true), np.unique(y_pred)]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale="Blues",
        x=labels,
        y=labels,
        labels=dict(x="Predicción", y="Real", color="Conteo"),
        title="Matriz de confusión",
    )
    fig.update_layout(height=500, margin=dict(l=60, r=20, t=60, b=60))
    return fig


def plot_roc_ovr(y_true, proba, classes):
    """
    Curvas ROC:
      - Binaria: usa proba de la clase positiva (columna 1).
      - Multiclase: one-vs-rest para cada clase.
    """
    classes = np.array(classes)

    # Caso binario claro
    if np.ndim(proba) == 2 and proba.shape[1] == 2:
        # Heurística: toma la clase "mayor" como positiva si no se pasan nombres
        y_true_bin = (y_true == classes.max()).astype(int)
        fpr, tpr, _ = roc_curve(y_true_bin, proba[:, 1])
        auc_val = auc(fpr, tpr)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {auc_val:.3f}"))
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="azar", line=dict(dash="dash"))
        )
        fig.update_layout(
            title="ROC",
            xaxis_title="FPR",
            yaxis_title="TPR",
            height=500,
            margin=dict(l=60, r=20, t=60, b=60),
        )
        return fig

    # Multiclase one-vs-rest
    y_bin = label_binarize(y_true, classes=classes)
    fig = go.Figure()
    for i, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{c} (AUC={auc(fpr, tpr):.3f})"))
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="azar", line=dict(dash="dash"))
    )
    fig.update_layout(
        title="ROC (One-vs-Rest)",
        xaxis_title="FPR",
        yaxis_title="TPR",
        height=520,
        margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig


def plot_pred_proba_hist(proba, positive_index=1):
    """Histograma de probabilidades (o scores si no hay probas)."""
    if np.ndim(proba) == 2 and proba.shape[1] >= 2:
        p = proba[:, positive_index]
        fig = px.histogram(
            p, nbins=40, title="Distribución de probabilidad (clase positiva)",
            labels={"value": "probabilidad", "count": "frecuencia"}
        )
    else:
        p = np.ravel(proba)
        fig = px.histogram(
            p, nbins=40, title="Distribución de score",
            labels={"value": "score", "count": "frecuencia"}
        )
    fig.update_layout(bargap=0.02, height=400, margin=dict(l=40, r=20, t=60, b=60))
    return fig


def tree_png_html(rf, feature_names, class_names=None, tree_index=0, max_depth=3):
    """
    Renderiza un árbol del RandomForest como <img> (PNG base64) para incrustar en HTML.

    rf            : modelo RandomForest
    feature_names : lista/Index de nombres de columnas
    class_names   : nombres de clases (clasificación) o None
    tree_index    : índice de árbol dentro del bosque
    max_depth     : profundidad máxima mostrada (mejora legibilidad)
    """
    # Importar Matplotlib solo cuando lo necesitamos
    plt = _mpl_setup()

    # sklearn.tree también dentro, por si el modelo no es RF en algunos deploys
    from sklearn import tree

    # Comprobar que hay estimators
    if not hasattr(rf, "estimators_") or len(rf.estimators_) == 0:
        # Si no hay árboles (por ejemplo, el modelo no es RF), devolvemos una etiqueta
        return '<div style="padding:8px;border:1px solid #ddd;border-radius:8px;">' \
               'No se pudo renderizar un árbol (modelo sin estimators).</div>'

    estimator = rf.estimators_[tree_index % len(rf.estimators_)]
    fig, ax = plt.subplots(figsize=(14, 8))
    tree.plot_tree(
        estimator,
        feature_names=list(feature_names),
        class_names=class_names,
        filled=True,
        impurity=True,
        rounded=True,
        max_depth=max_depth,
        fontsize=8,
        ax=ax,
    )
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f'<img src="data:image/png;base64,{data}" alt="Árbol {tree_index}" />'


__all__ = [
    "fig_to_html",
    "plot_feature_importance",
    "plot_confusion",
    "plot_roc_ovr",
    "plot_pred_proba_hist",
    "tree_png_html",
]
