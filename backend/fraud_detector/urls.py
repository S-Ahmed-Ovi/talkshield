from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.fraud_predict, name='fraud_predict'),
]
