from django.urls import path, re_path, include
from .views import CurrentPositionView,SignsView, SignsList, ToUpdateSignsList, FilteringView
from rest_framework.authtoken.views import obtain_auth_token

urlpatterns = [
    path('position/',CurrentPositionView.as_view()),
    path('api-token-auth/', obtain_auth_token, name='api_token_auth'),
    path('signs/<int:pk>/', SignsView.as_view()),
    path('signs/list',SignsList.as_view()),
    path('signs/toupdate',ToUpdateSignsList.as_view()),
    path('signs/query',FilteringView.as_view()),

]
