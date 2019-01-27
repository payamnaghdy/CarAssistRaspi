from django.shortcuts import render
from rest_framework import generics
from .serializer import CurrentPositionSerializer, SignsSerializer
from .models import CurrentPosition, Signs
from rest_framework.permissions import IsAuthenticated
from django.http import Http404
from rest_framework.response import Response
from django.http import HttpResponseNotFound
# Create your views here.

class CurrentPositionView(generics.ListAPIView):
    queryset=CurrentPosition.objects.filter(pk=1)
    serializer_class = CurrentPositionSerializer
    permission_classes = (IsAuthenticated,)
    http_method_names = ['get','put']

    def put(self,request):
        position=CurrentPosition.objects.filter(pk=1).first()
        print(request.data)
        serializer =  CurrentPositionSerializer(position,data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)


class SignsView(generics.ListAPIView):
    serializer_class = SignsSerializer
    permission_classes= (IsAuthenticated,)
    http_method_names = ['get','put']
    def get(self, request, pk):
        try:
            print(pk)
            serializer = SignsSerializer(Signs.objects.filter(pk=pk).first())
            return Response(serializer.data)
        except Exception as e:
             return HttpResponseNotFound("404") 
    def put(self,request,pk):
        sign=Signs.objects.filter(pk=pk).first()
        serializer =  SignsSerializer(sign,data=request.data)
        if serializer.is_valid():
            serializer.save()
            return 'done'

class ToUpdateSignsList(generics.ListAPIView):
    serializer_class =SignsSerializer
    permission_classes=(IsAuthenticated,)
    http_method_names=['get']
    queryset=Signs.objects.filter(is_uploaded=False)
    # def get(self,request,format=None):
    #     serializer = SignsSerializer(self.queryset, many=True)
    #     return Response(serializer.data)

class SignsList(generics.ListAPIView):
    serializer_class = SignsSerializer
    permission_classes= (IsAuthenticated,)
    http_method_names = ['get','post']
    queryset=Signs.objects.all()
    def post(self, request, format=None):
        serializer = SignsSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return HttpResponseNotFound("201") 
        return HttpResponseNotFound("400") 
class FilteringView(generics.ListAPIView):
    serializer_class=SignsSerializer
    serializer_class = SignsSerializer
    #permission_classes= (IsAuthenticated,)
    http_method_names = ['get']
    queryset=Signs.objects.all()
    def get(self,request):
        country = request.query_params['country']
        if country is not None :
            qset=Signs.objects.all().filter(country=country)
        county = request.query_params['country']
        if county is not None :
            qset=Signs.objects.all().filter(county=county)
        neighbourhood = request.query_params['neighbourhood']
        if neighbourhood is not None :
            neighbourhood=Signs.objects.all().filter(neighbourhood=neighbourhood)
        road = request.query_params['road']
        if road is not None :
            road=Signs.objects.all().filter(road=road)

        # county=request.query_params('county',None)
        # neighbourhood=request.query_params('neighbourhood',None)
        # road=request.query_params('road',None)
        
        serializer=SignsSerializer(qset,many=True)
        return Response(serializer.data)
        