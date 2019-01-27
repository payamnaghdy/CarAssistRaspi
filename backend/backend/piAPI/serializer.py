from rest_framework import serializers
from .models import Signs, CurrentPosition
from rest_framework.validators import UniqueTogetherValidator

class SignsSerializer(serializers.ModelSerializer):
    class Meta:
        model=Signs
        fields = '__all__'
        validators = [
            UniqueTogetherValidator(
                queryset=Signs.objects.all(),
                fields=('name', 'country','county','neighbourhood','road' )
            )]



class CurrentPositionSerializer(serializers.ModelSerializer):
    class Meta:
        model=CurrentPosition
        fields = '__all__'