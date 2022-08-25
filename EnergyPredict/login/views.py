from cgitb import html
from django.shortcuts import render

from login import models
from login import tests

def login(request):
# 看是否已经登录已经登录则跳转到predict
    try:
        cookie = request.COOKIES['Form1ng']
        SessionCookie = request.session['IsLogined']
        if cookie == SessionCookie:
            return render(request,'predict.html',content_type="text/html")
    except:
# 登录请求还是一个访问请求
        if request.method == 'GET':
            return render(request,'login.html')
        else:
            try:
                usern = request.POST['username']
                passw = request.POST['password']
                if models.UserInfo.objects.filter(username=usern).values_list('password',flat=True).count() > 0:
                    cookie = tests.EnCookie(usern,passw)
                    request.session['IsLogined'] = cookie
                    response = render(request,'predict.html',content_type="text/html")
                    response.set_cookie("Form1ng",cookie,max_age=86400)
                    return response
                else:
                    return render(request,'login.html')
            except:
                return render(request,'login.html')