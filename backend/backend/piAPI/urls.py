from django.urls import path, re_path, include
from .views import CurrentPositionView

urlpatterns = [
    path('position/',CurrentPositionView.as_view()),


]