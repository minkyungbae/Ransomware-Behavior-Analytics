from django.urls import path
from . import views

urlpatterns = [
    path("", views.mainpage, name="mainpage"),
    path("ping/", views.ping),
    path("train/", views.train_models),
    path("samples/", views.get_samples),
]
