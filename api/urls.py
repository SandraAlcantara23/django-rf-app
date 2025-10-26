# api/urls.py
from django.urls import path
from .views import home, predict_csv, health, api_predict, train_page

urlpatterns = [
    path("", home, name="home"),
    path("predict/csv/", predict_csv, name="predict_csv"),
    path("health/", health, name="health"),
    path("api/predict/", api_predict, name="api_predict"),
    path("train/", train_page, name="train_page"),
]
