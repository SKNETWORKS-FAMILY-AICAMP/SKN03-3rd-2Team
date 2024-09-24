from django.urls import path
from .views import main_view, more_view

urlpatterns = [
    path("", main_view, name="prediction"),
    path("more/", more_view, name="more_prediction"),
]
