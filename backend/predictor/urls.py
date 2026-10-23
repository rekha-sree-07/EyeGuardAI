
from django.urls import path
from .views import predict_eye_disease, predict_batch

urlpatterns = [
    path("predict/", predict_eye_disease, name="predict_eye_disease"),
    path("predict/batch/", predict_batch, name="predict_batch"),
]
