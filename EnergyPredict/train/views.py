# django自带模块
from cProfile import label
from cgitb import handler
from importlib.resources import path
from itertools import count
from logging import handlers
from re import T
from tkinter import Y
from turtle import color
from django.shortcuts import render
from flask import request
from markupsafe import re
#自己写的模块
from train.tests import getImageName
# 画图所用模块
import os
import matplotlib.pyplot as plt
from matplotlib import ticker
#机器学习需要的模块
import torch
from torch import nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
#文件管理及命令执行
import os
import random

# Create your views here.
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

def train(request):
    try:
        if str(request.session['IsLogined']) == str(request.COOKIES['Form1ng']):
            try:
                if request.method == 'POST':
                    # 下面的代码是读取数据并转换为合适的数据类型的代码
                    # 判断是使用默认的train和test还是新的
                    try:
                        if 'file' in request.FILES:
                            csvfilename = str(random.random())[0:10] + '.csv'
                            csvfilepath = 'train/upload/' + csvfilename
                            with open(csvfilepath,'wb+') as f:
                                for chunk in request.FILES['file'].chunks():
                                    f.wirte(chunk)
                            data = pd.read_csv(csvfilepath)
                            csvtrainjudge = False
                        else:
                            csvfilename = 'E_n_m_dataset.csv'
                            csvfilepath = 'train/E_n_m_dataset.csv'
                            data = pd.read_csv("train/E_n_m_dataset.csv")
                            csvtrainjudge = True
                    except:
                        data = pd.read_csv("train/E_n_m_dataset.csv")
                        csvtrainjudge = True
                    # 根据情况指定x，y的维度
                    try:
                        if request.POST['xsize'] == 12 and request.POST['ysize'] == 3:
                            xsize = 12
                            ysize = 3
                            totalsize = xsize + ysize
                            sizetrainjudge = True
                        else:
                            xsize = int(request.POST['xsize'])
                            ysize = int(request.POST['ysize'])
                            totalsize = xsize + ysize
                            sizetrainjudge = False
                    except:
                        xsize = 12
                        ysize = 3
                        totalsize = xsize + ysize
                        sizetrainjudge = True
                    # 获取训练出的模型的文件名，或者获取已有模型的文件名
                    try:
                        if str(request.POST['ptfilename']).split('.')[0] == 'maml5Layer' and str(request.POST['ptfilename']).split('.')[1] == 'pt' and len(str(request.POST['ptfilename']).split('.')) == 2 and not re.search(';|&|\|',str(request.POST['modelname']).split('.')[0]):
                            ptfilename = 'maml5LayerRelu.pt'
                            pttrainjudge = True
                        elif str(request.POST['ptfilename']).split('.')[0] != 'maml5Layer' and str(request.POST['ptfilename']).split('.')[1] == 'pt' and os.path.exists('train/upload/' + request.POST['ptfilename']) and len(str(request.POST['ptfilename']).split('.')) == 2 and not re.search(';|&|\|',str(request.POST['modelname']).split('.')[0]):
                            ptfilename = request.POST['ptfilename']
                            pttrainjudge = True
                        elif str(request.POST['ptfilename']).split('.')[0] != 'maml5Layer' and str(request.POST['ptfilename']).split('.')[1] == 'pt' and not (os.path.exists('train/upload/' + request.POST['ptfilename'])) and len(str(request.POST['ptfilename']).split('.')) == 2 and not re.search(';|&|\|',str(request.POST['modelname']).split('.')[0]):
                            ptfilename = request.POST['ptfilename']
                            pttrainjudge = False
                        else:
                            ptfilename = 'maml5LayerRelu.pt'
                            pttrainjudge = True
                    except:
                        ptfilename = 'maml5LayerRelu.pt'
                        pttrainjudge = True
                    #提取并转换test
                    if pttrainjudge or sizetrainjudge or csvtrainjudge:
                        source_data = data.iloc[:, :xsize]
                        source_label = data.iloc[:, xsize:totalsize]
                        source_data = np.array(source_data)
                        source_label = np.array (source_label)
                        train_size = int (len(source_data) * 0.75)
                        test_size = len(source_data) - train_size
                        x_train, x_test= random_split(source_data, [train_size, test_size])
                        x_test = torch.FloatTensor(x_test)
                        y_train, y_test = random_split(source_label, [train_size, test_size])
                        # y_train = torch.FloatTensor(y_train)
                        y_test = torch.FloatTensor(y_test)
                        # 通过GetLoader将数据进行加载，返回Dataset对象，包含data和labels
                        test_data = GetLoader(x_test, y_test)
                        # 读取数据
                        test_datas = DataLoader(test_data, batch_size=4, shuffle=True, drop_last=False)
                        test_x = []
                        test_label = []
                        for x, y in test_datas:
                            for i in range(len(x)):
                                test_x.append(x[i])
                                test_label.append(y[i])
                        test_x = torch.cat(test_x, dim=0)
                        test_x = test_x.reshape((test_size, xsize))
                        test_label = torch.cat(test_label, dim=0)
                        test_label = test_label.reshape((test_size, ysize))
                        # 下面的代码是利用上面处理完的数据跑meta-learning
                        transform = StandardScaler()
                        # train_x = transform.fit_transform(train_x)
                        test_x = transform.fit_transform(test_x)
                        # train_x = torch.FloatTensor(train_x)
                        test_x = torch.FloatTensor(test_x)
                        # 这里已经不需要重构模型结构了，直接load就可以
                        maml = torch.load('train/upload/' + ptfilename)
                        #下面决定使用的预测
                        Y_test, Y_predict = maml(test_x, test_label)
                        # 生成保存图片的文件名
                        filename = getImageName()
                        filepathContrast = '/static/PredictImage/' + filename + 'Contrast.png'
                        filepathLoss = '/static/PredictImage/' + filename + 'Loss.png'
                        imagepathContrast = filename + 'Contrast.png'
                        imagepathLoss = filename + 'Loss.png'
                        saveimageContrast = 'static/PredictImage/' + filename + 'Contrast.png'
                        saveimageLoss = 'static/PredictImage/' + filename + 'Loss.png'
                        #用plt画图并保存（对比图）
                        plt.figure()
                        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
                        if xsize == 12 and ysize == 3:
                            plt.scatter(range(len(Y_predict)), Y_predict.detach()[:,0],color = 'red',marker='^', label='y1_predict')
                            plt.scatter(range(len(Y_predict)), Y_test.detach()[:,0],color='green',marker='^', label='y1_test')
                            plt.scatter(range(len(Y_predict)), Y_predict.detach()[:,1],color='blue',marker='^', label='y2_predict')
                            plt.scatter(range(len(Y_predict)), Y_test.detach()[:,1],color='yellow',marker='^', label='y2_test')
                            plt.scatter(range(len(Y_predict)), Y_predict.detach()[:,2],color='black',marker='^', label='y3_predict')
                            plt.scatter(range(len(Y_predict)), Y_test.detach()[:,2],color='pink',marker='^', label='y3_test')
                            plt.legend(prop='chinese')
                        else:
                            for i in range(len(Y_predict)):
                                plt.scatter(range(len(Y_predict)),Y_predict.detach()[:,i],color='red',marker='^', label='y'+ str(i) +"predict")
                                plt.scatter(range(len(Y_predict)), Y_test.detach()[:,i],color='green',marker='^', label='y'+ str(i) + 'test')
                                plt.legend(prop='chinese')
                        plt.xlabel("Count of predict result")
                        plt.ylabel("value of sales")
                        plt.savefig(saveimageContrast)
                        plt.close()
                        #重绘plt图片并保存（loss图）
                        plt.figure()
                        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
                        Form1ng = []
                        for b in range(0,len(Y_predict)):
                            for c in range(ysize):
                                Form1ng.append(abs(Y_predict[b][c].detach().item() - Y_test[b][c].detach().item()))
                        result = []
                        middle = 0.0
                        for d in range(test_size):
                            for e in range(ysize):
                                middle = middle + Form1ng[d*3+e]
                            result.append(middle)
                            middle = 0.0
                        plt.plot(range(len(Y_predict)), result, 'ro', label='abs')
                        plt.xlabel("Count of loss function result")
                        plt.ylabel("value of sales")
                        plt.savefig(saveimageLoss)
                        plt.close()
                        if csvfilepath != 'train/E_n_m_dataset.csv':
                            os.remove(csvfilepath)
                        return render(request,'train.html',{'judge':True,'JUDGE':False,'Judge':False,'filepathContrast':filepathContrast,'filepathLoss':filepathLoss,'imagepathContrast':imagepathContrast,'imagepathLoss':imagepathLoss})
                    else:
                        commonand = 'python ./main.py --xsize ' + xsize + ' --ysize ' + ysize + ' --ptfilename ' + ptfilename + ' --csvfilename ' + csvfilename
                        os.system(commonand)
                        return render(request,'train.html',{'judge':False,'JUDGE':False,'Judge':True})
                else:
                    return render(request,'train.html')
            except:
                return render(request,'train.html',{'judge':False,'JUDGE':True,'Judge':False})
        else:
            return render(request,'login.html')
    except Exception:
        return render(request,'login.html')