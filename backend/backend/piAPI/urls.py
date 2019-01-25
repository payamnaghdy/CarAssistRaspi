from django.urls import path, re_path, include
from .views import CurrentPositionView
from rest_framework.authtoken.views import obtain_auth_token

urlpatterns = [
    path('position/',CurrentPositionView.as_view()),
    path('api-token-auth/', obtain_auth_token, name='api_token_auth'),

]