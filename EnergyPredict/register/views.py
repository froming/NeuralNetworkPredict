from asyncio.windows_events import NULL
from django.http import HttpResponse
from django.shortcuts import render
import random
from datetime import datetime

from login import models as database
from register import tests

def register(request):
    if request.method == 'POST':
# 检验验证码是否过期
        FirstTime = request.session['time']
        if tests.JudgeDateMiss(FirstTime,240):
            judge = ''
            for i in range(4):
                i = str(random.randint(0,9))
                judge = judge + i
            request.session['time'] = tests.GetNowDateList()
            request.session['Vcode'] = str(judge)
            print(1)
            return render(request,'register.html',{'judge':judge})
# 检测输入的验证码是否正确，并向数据库中插入数据
        else:
            judge = request.session['Vcode']
            if str(request.POST['Vcode']) == str(judge):
                Form1ng = request.POST['Form1ng']
                form1ng = request.POST['form1ng']
                if Form1ng != '' or form1ng != '':
                    return render(request,'register.html',{'judge':judge})
                usern = request.POST['username']
                passw = request.POST['password']
                database.UserInfo.objects.create(username=usern,password=passw)
                return render(request,'login.html')
            else:
                print(2)
                return render(request, 'register.html',{'judge':judge})
    else:
# 生成验证码并插入数据库
        judge = ''
        for i in range(4):
            i = str(random.randint(0,9))
            judge = judge + i
        request.session['Vcode'] = str(judge)
#time_first是第一次get登录时生成验证码的时间用于验证验证码是否过期
        TimeFirst = str(datetime.now())
        request.session['time'] = tests.GetDateList(TimeFirst)
        return render(request,'register.html',{'judge':judge})