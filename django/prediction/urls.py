
from django.urls import path
from .views import prediction

urlpatterns = [
    path('', prediction, name="prediction"),
]
