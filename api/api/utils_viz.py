# api/utils_viz.py
import io
import base64
import numpy as np

# Plotly (interactivo, muy liviano)
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

def fig_to_html(fig):
    """HTML embebible de Plotly (incluye plotly.js vía CDN)."""
    return fig.to_html(full_html=False, include_plotlyjs="cdn")

def plot_feature_importance(model, feature_names, top=20):
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        importances = np.zeros(len(feature_names))
    order = np.argsort(importances)[::-1][:top]
    fig = px.bar(
        x=importances[order][::-1],
        y=np.array(list(feature_names))[order][::-1],
        orientation="h",
        title=f"Importancia de características (top {top})",
        labels={"x": "Importancia", "y": "Feature"},
    )
    fig.update_layout(height=520, margin=dict(l=90, r=20, t=60, b=40))
    return fig

def plot_confusion(y_true, y_pred):
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
    classes = np.array(classes)

    # Caso binario
    if proba.ndim == 2 and proba.shape[1] == 2:
        # asumimos la clase mayor como positiva si no se proveen names
        pos = classes.max()
        y_true_bin = (y_true == pos).astype(int)
        fpr, tpr, _ = roc_curve(y_true_bin, proba[:, 1])
        auc_val = auc(fpr, tpr)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC={auc_val:.3f}"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="azar", line=dict(dash="dash")))
        fig.update_layout(title="ROC", xaxis_title="FPR", yaxis_title="TPR", height=500)
        return fig

    # Multiclase One-vs-Rest
    y_bin = label_binarize(y_true, classes=classes)
    fig = go.Figure()
    for i, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
        fig.add_trace(go.Scatter(x={0: fpr, 1: tpr}[0], y={0: fpr, 1: tpr}[1], mode="lines",
                                 name=f"{c} (AUC={auc(fpr, tpr):.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="azar", line=dict(dash="dash")))
    fig.update_layout(title="ROC (One-vs-Rest)", xaxis_title="FPR", yaxis_title="TPR", height=500)
    return fig

def plot_pred_proba_hist(proba, positive_index=1):
    if proba.ndim == 2 and proba.shape[1] >= 2:
        p = proba[:, positive_index]
        fig = px.histogram(p, nbins=40, title="Distribución de probabilidad (clase positiva)",
                           labels={"value": "probabilidad", "count": "frecuencia"})
    else:
        p = proba.ravel()
        fig = px.histogram(p, nbins=40, title="Distribución de score",
                           labels={"value": "score", "count": "frecuencia"})
    fig.update_layout(bargap=0.02, height=400, margin=dict(l=40, r=20, t=60, b=40))
    return fig

def tree_png_html(rf, feature_names, class_names=None, tree_index=0, max_depth=3):
    """
    Dibuja un árbol del RandomForest como <img> PNG (base64).
    **Importa matplotlib solo aquí** para evitar coste en el arranque.
    """
    import os
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn import tree as sktree

    estimator = rf.estimators_[tree_index]
    fig, ax = plt.subplots(figsize=(14, 8))
    sktree.plot_tree(
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
