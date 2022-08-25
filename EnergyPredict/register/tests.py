from django.test import TestCase

import datetime
# Create your tests here.
# 获取当前datetime的list格式
def GetNowDateList():
    first = datetime.datetime.now()
    FirstStr = str(first).split('.')[0]
    YMD = FirstStr.split(' ')[0]
    TIME = FirstStr.split(' ')[1]
    YMDed = YMD.split('-')
    TIMEed = TIME.split(':')
    final = YMDed + TIMEed
    return final

# 获取某一datetime的list格式
def GetDateList(DateTime):
    FirstStr = str(DateTime).split('.')[0]
    YMD = FirstStr.split(' ')[0]
    TIME = FirstStr.split(' ')[1]
    YMDed = YMD.split('-')
    TIMEed = TIME.split(':')
    final = YMDed + TIMEed
    return final

# 判断某一时间与当前时间的时间间隔以秒为单位
def JudgeDateMiss(FirstDatelist,standard):
    SecondDateList = GetNowDateList()
    if int(SecondDateList[0]) > int(FirstDatelist[0]):
        return True
    if int(SecondDateList[1]) > int(FirstDatelist[1]):
        return True
    if int(SecondDateList[2]) > int(FirstDatelist[2]):
        return True
    if int(SecondDateList[3]) > int(FirstDatelist[3]):
        return True
    if (int(SecondDateList[4]) - int(FirstDatelist[4]))*60 + int(SecondDateList[5]) > standard :
        return True
    return False
