from django.db import models

class UserInfo(models.Model):
    username = models.CharField(max_length=32,default="")
    password = models.CharField(max_length=32,default="")