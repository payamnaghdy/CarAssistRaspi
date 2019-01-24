from django.shortcuts import render
from rest_framework import generics
from .serializer import CurrentPositionSerializer, SignsSerializer
from .models import CurrentPosition
from rest_framework.permissions import IsAuthenticated
# Create your views here.

class CurrentPositionView(generics.ListAPIView):
    queryset=CurrentPosition.objects.filter(pk=1)
    serializer_class = CurrentPositionSerializer
    permission_classes = (IsAuthenticated,)
    http_method_names = ['get','put']
   
    def put(self,request):
        position=queryset.first()
        serializer =  CurrentPositionSerializer(position,data=request.data)
        if serializer.is_valid():
            serializer.save()
            return 'done'