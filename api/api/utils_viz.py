import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

def fig_to_html(fig):
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def plot_feature_importance(model, feature_names, top=20):
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        importances = np.zeros(len(feature_names))
    order = np.argsort(importances)[::-1][:top]
    fig = px.bar(
        x=importances[order][::-1],
        y=np.array(feature_names)[order][::-1],
        orientation='h',
        title=f"Importancia de características (top {top})",
        labels={'x': 'Importancia', 'y': 'Feature'}
    )
    return fig

def plot_confusion(y_true, y_pred):
    labels = np.unique(np.concatenate([np.unique(y_true), np.unique(y_pred)]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = px.imshow(
        cm, text_auto=True, color_continuous_scale="Blues",
        x=labels, y=labels,
        labels=dict(x="Predicción", y="Real", color="Conteo"),
        title="Matriz de confusión"
    )
    return fig
