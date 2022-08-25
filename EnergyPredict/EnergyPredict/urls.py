"""EnergyPredict URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from random import triangular
from django.contrib import admin
from django.urls import path

from login.views import login
from register.views import register
from predict.views import predict
from train.views import train
from . import views
from download.views import download

urlpatterns = [
    path('admin/', admin.site.urls),
    path('index/', login),
    path('login/',login),
    path('register/', register),
    path('predict/', predict),
    path('train/', train),
    path('download/', download),
    path('', login),
]

handler = views.page_not_found