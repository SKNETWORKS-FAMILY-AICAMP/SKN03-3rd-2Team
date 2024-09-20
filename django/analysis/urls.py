
from django.urls import path
from .views import analysis

urlpatterns = [
    path('', analysis, name="analysis"),
]
