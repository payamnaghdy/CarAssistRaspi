from rest_framework import serializers
from .models import Signs, CurrentPosition

class SignsSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model=Signs
        feilds='__all__'




class CurrentPositionSerializer(serializers.ModelSerializer):
    class Meta:
        model=CurrentPosition
        fields = '__all__'