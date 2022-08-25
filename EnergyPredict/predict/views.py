# django自导入模块
from django.shortcuts import render
# 数据库导入
from predict import models
# 机器学习需要的模块
import os
import torch
from torch import nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

import re

class MamlModel(nn.Module):
    def __init__(self, input_dim, out_dim, x_train, hidden_units):
        super(MamlModel, self).__init__()
        self.X_train = x_train
        self.linear1 = nn.Linear(input_dim, hidden_units)
        self.linear2 = nn.Linear(hidden_units, hidden_units)
        self.linear3 = nn.Linear(hidden_units, hidden_units)
        self.linear4 = nn.Linear(hidden_units, hidden_units)
        self.linear5 = nn.Linear(hidden_units, out_dim)
    def forward(self, input, input_label):
        # 单层网络的训练
        x = self.linear1(input)
        x = torch.tanh(x)
        x = self.linear2(x)
        x = torch.relu(x)
        x = self.linear3(x)
        x = torch.tanh(x)
        x = self.linear4(x)
        x = torch.relu(x)
        Y_predict = self.linear5(x)
        return input_label, Y_predict

# 定义GetLoader类，继承Dataset方法，并重写__getitem__()和__len__()方法
class GetLoader(torch.utils.data.Dataset):
    # 初始化函数，得到数据
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label
    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels
    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)

# 异氰酸酯分子量或者多元醇分子量
def effectmol(componentVol,componentdensity,weight):
    return (componentVol*componentdensity)/weight

# 官能团分子量
def groupmol(componentVol,componentdensity,weight,componentgroup):
    return (componentVol*componentdensity)/weight*componentgroup

# 官能团重量，异氰酸酯为160.1，羟基为17
def groupWeight(componentVol,componentdensity,weight,componentgroup,groupWeight):
    return (componentVol*componentdensity)/weight*componentgroup*groupWeight

# 链长
def chainLength(adMol,jtCount):
    return adMol/jtCount

# R值
def RValue(NCOmol,OHmol,hetepercent,adpercent):
    return (NCOmol*hetepercent)/(OHmol*adpercent)

# 聚合物质量
def totalWeight(heteDensity,heteVol,adDensity,adVol):
    return heteDensity*heteVol+adDensity+adVol

# 硬连接
def hardLink(heteVol,heteDensity,RValue,totalWeight):
    return (heteVol*heteDensity*RValue)/totalWeight

# 软连接
def softLink(hardLink):
    return 1-hardLink

def predict(request):
    try:
        if str(request.session['IsLogined']) == str(request.COOKIES['Form1ng']):
            try:
                if request.method == 'POST':
# 下面的代码是读取数据并转换为合适的数据类型的代码
                    # 提取并转换test
                    try:
                        if request.POST['judge'] == 'predict':
                            x_test = []
                            y_test = [0.38275,-0.00931,0.39]
                            # 通过数据库获取其它相关数据
                            componentTwo = float(request.POST['componentTwo'])
                            componentOne = request.POST['componentOne']
                            temprature = request.POST['temprature']
                            componentOneList = models.ComponentParameter.objects.filter(ComponentName=componentOne,temprature=temprature)
                            componentTwoList = models.ComponentParameter.objects.filter(ComponentName=componentTwo,temprature=temprature)
                            # 底层数据
                            # 异固化剂的的底层数据
                            componentOneDensity = componentOneList.values('density')
                            componentOneWeight = componentOneList.values('weight')
                            componentOnegroup = componentOneList.values('group')
                            percentOne = float(request.POST['percentOne'])
                            componentOneVol = float(request.POST['NCOvol'])
                            # 粘合剂的底层数据
                            componentTwoDensity = componentTwoList.values('density')
                            componentTwoWeight = componentTwoList.values('weight')
                            componentTwogroup = componentTwoList.values('group')
                            percentTwo = float(request.POST['percentTwo'])
                            componentTwoVol = float(request.POST['OHvol'])
                            # 代入模型的名称
                            if len(str(request.POST['modelname']).split('.')) == 2 and str(request.POST['modelname']).split('.')[1] == 'pt' and not re.search(';|&|\|',str(request.POST['modelname']).split('.')[0]):
                                modelname = request.POST['modelname']
                            # 这里是人工或者脚本跑出得，但是是可以直接用得
                            CO_W = float(request.POST['CO_W'])
                            NH_W = float(request.POST['NH_W'])
                            NH_A = float(request.POST['NH_A'])
                            CED = float(request.POST['CED'])
                            solubility = float(request.POST['solubility'])
                            core_pctWgt = float(request.POST['core_pctWgt'])
                            sol_pctWgt = float(request.POST['sol_pctWgt'])
                            Mtw = float(request.POST['Mtw'])
                            Form1ng = float(request.POST['Form1ng'])
                            benhuan = float(request.POST['benhuan'])
                            # 由部分基础数据得出的中间层数据
                            totalweight = totalWeight(componentOneDensity,componentOneVol,componentTwoDensity,componentTwoVol)
                            NCOmol = groupmol(componentOneVol,componentOneDensity,componentOneWeight,componentOnegroup)
                            OHmol = groupmol(componentTwoVol,componentTwoDensity,componentTwoWeight,componentTwogroup)
                            rValue = RValue(NCOmol,OHmol,percentOne,percentTwo)
                            # 最终在模型中要用到得两个变量也得到了，下面直接引入模型就能跑
                            hardlink = hardLink(componentOneVol,componentOneDensity,rValue,totalweight)
                            softlink = softLink(hardlink)
                            # 12变量
                            x1,x2,x3 = CO_W,NH_W,NH_A
                            x4,x5,x6,x7 = softlink,hardlink,Form1ng,benhuan
                            x8,x9,x10,x11,x12 = CED,solubility,core_pctWgt,sol_pctWgt,Mtw
                            x_test.append(x1)
                            x_test.append(x2)
                            x_test.append(x3)
                            x_test.append(x4)
                            x_test.append(x5)
                            x_test.append(x6)
                            x_test.append(x7)
                            x_test.append(x8)
                            x_test.append(x9)
                            x_test.append(x10)
                            x_test.append(x11)
                            x_test.append(x12)
                            x_test = [x_test,x_test,x_test,x_test,x_test]
                            y_test = [y_test,y_test,y_test,y_test,y_test]
                            x_test = np.array(x_test)
                            y_test = np.array(y_test)
                            x_test = torch.FloatTensor(x_test)
                            y_test = torch.FloatTensor(y_test)
                            test_data = GetLoader(x_test,y_test)
                            test_datas = DataLoader(test_data, batch_size=4, shuffle=True, drop_last=False)
                            test_x = []
                            test_label = []
                            for x,y in test_datas:
                                for i in range(len(x)):
                                    test_x.append(x[i])
                                    test_label.append(y[i])
                            test_x = torch.cat(test_x, dim=0)
                            test_x = test_x.reshape((5, 12))
                            test_label = torch.cat(test_label, dim=0)
                            test_label = test_label.reshape((5, 3))
                            # 下面的代码是利用上面处理完的数据跑神经网络
                            transform = StandardScaler()
                            test_x = transform.fit_transform(test_x)
                            test_x = torch.FloatTensor(test_x)
                            # 这里已经不需要重构模型结构了，直接load就可以
                            maml = torch.load('train/upload/' + modelname)
                            #下面决定使用的预测
                            Y_test, Y_predict = maml(test_x, test_label)
                            return render(request,'predict.html',{'judge':True,'JUDGE':False,'strain':Y_predict[0][0].detach().item(),'stress':Y_predict[0][1].detach().item(),'tan':Y_predict[0][2].detach().item()})
                        if request.POST['judge'] == 'jisuan':
                            # 通过数据库获取其它相关数据
                            componentTwo = float(request.POST['componentTwo'])
                            componentOne = request.POST['componentOne']
                            temprature = request.POST['temprature']
                            componentOneList = models.ComponentParameter.objects.filter(ComponentName=componentOne,temprature=temprature)
                            componentTwoList = models.ComponentParameter.objects.filter(ComponentName=componentTwo,temprature=temprature)
                            # 底层数据
                            # 异固化剂的的底层数据
                            componentOneDensity = componentOneList.values('density')
                            componentOneWeight = componentOneList.values('weight')
                            componentOnegroup = componentOneList.values('group')
                            percentOne = float(request.POST['percentOne'])
                            componentOneVol = float(request.POST['NCOvol'])
                            # 粘合剂的底层数据
                            componentTwoDensity = componentTwoList.values('density')
                            componentTwoWeight = componentTwoList.values('weight')
                            componentTwogroup = componentTwoList.values('group')
                            percentTwo = float(request.POST['percentTwo'])
                            componentTwoVol = float(request.POST['OHvol'])
                            # 基团分子量
                            jtCount = float(request.POST['jtCount'])
                            # 由部分基础数据得出的中间层数据
                            heteMol = effectmol(componentOneVol,componentOneDensity,componentOneWeight)
                            adMol = effectmol(componentTwoVol,componentTwoDensity,componentTwoWeight)
                            NCOmol = groupmol(componentOneVol,componentOneDensity,componentOneWeight,componentOnegroup)
                            OHmol = groupmol(componentTwoVol,componentTwoDensity,componentTwoWeight,componentTwogroup)
                            NCOweight = groupWeight(componentOneVol,componentOneDensity,componentOneWeight,componentOnegroup,160.1)
                            OHweight = groupWeight(componentTwoVol,componentTwoDensity,componentTwoWeight,componentTwogroup,17)
                            totalweight = totalWeight(componentOneDensity,componentOneVol,componentTwoDensity,componentTwoVol)
                            rValue = RValue(NCOmol,OHmol,percentOne,percentTwo)
                            hardlink = hardLink(componentOneVol,componentOneDensity,rValue,totalweight)
                            softlink = softLink(hardlink)
                            ChainLength = chainLength(adMol,jtCount)
                            return render(request,'predict.html',{'judge':False,'JUDGE':True,'hetemol':heteMol,'admol':adMol,'NCOmol':NCOmol,'OH':OHmol,'chain':ChainLength,'R':rValue,'totalweight':totalweight,'hard':hardlink,'soft':softlink,'OHweight':OHweight,'NCOweight':NCOweight})
                    except:
                        return render(request,'predict.html',{'judge':True})
                else:
                    return render(request,'predict.html',{'judge':True})
            except:
                return render(request,'predict.html',{'judge':True})
        else:
            return render(request,'login.html')
    except Exception:
        return render(request,'login.html')