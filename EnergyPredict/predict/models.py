from email.policy import default
from django.db import models

# Create your models here.

class ComponentParameter(models.Model):
    #组分名字
    ComponentName = models.CharField(max_length=32,default="")
    #温度
    temperature = models.CharField(max_length=32,default="")
    # 密度
    density = models.FloatField(default=0)
    # 相对分子质量
    weight = models.FloatField(default=0)
    # 该有效分子一个分子有多少个官能团
    group = models.FloatField(default=0)