from django.urls import path
from .views import SpamPredictAPIView

urlpatterns = [
    path('predict/', SpamPredictAPIView.as_view(), name='spam-predict'),
]