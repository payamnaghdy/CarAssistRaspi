from django.db import models

# Create your models here.
class CurrentPosition(models.Model):
    latitude = models.FloatField()
    longitude=models.FloatField()
    country=models.CharField(max_length=50)
    county=models.CharField(max_length=50)
    neighbourhood=models.CharField(max_length=50)
    road=models.CharField(max_length=50)




class Signs(models.Model):
    name = models.CharField(max_length=200)
    latitude = models.FloatField()
    longitude=models.FloatField()
    country=models.CharField(max_length=50)
    county=models.CharField(max_length=50)
    neighbourhood=models.CharField(max_length=50)
    road=models.CharField(max_length=50)
    speedlimit=models.IntegerField()
    is_uploaded=models.BooleanField(default=False)

