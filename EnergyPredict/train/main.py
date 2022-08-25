import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pandas as pd

import argparse
import os
import re

# 获取参数
parser = argparse.ArgumentParser(description='Form1ng is handsome')
parser.add_argument('--xsize',type=int,default=12)
parser.add_argument('--ysize',type=int,default=3)
parser.add_argument('--ptfilename',default='maml5LayerRelu.pt')
parser.add_argument('--csvfilename',default='E_n_m_dataset.csv')
Form1ng = parser.parse_args()
# 生成文件的保存及获取路径
csvfilepath = 'upload/' + Form1ng.csvfilename
ptfilepath = 'upload/' + Form1ng.ptfilename
deffilename = str(Form1ng.ptfilename).split('.')[0] + '.txt'
desfilepath = 'upload/' + str(Form1ng.ptfilename).split('.')[0] + '.txt'
# 导入数据
data = pd.read_csv(csvfilepath)
# 删除csv文件
# os.remove(csvfilepath)

with open(desfilepath,'w') as f:
    f.write('the input number of model is ' + str(Form1ng.xsize) + '\n')
    f.write('the output number of model is ' + str(Form1ng.ysize) + '\n')
    f.write('the Layer is 5' + '\n')
    f.write('the dimesion of middlelayer is 8' + '\n')

# forming = open(desfilepath,'wb+')
source_data = data.iloc[:, :Form1ng.xsize]
source_label = data.iloc[:, Form1ng.xsize:Form1ng.xsize + Form1ng.ysize]
source_data = np.array(source_data)
source_label = np.array(source_label)
train_size = int (len(source_data) * 0.75)
test_size = len(source_data) - train_size
x_train, x_test= random_split(source_data, [train_size, test_size])
with open(desfilepath,'w') as f:
    f.write('the train x array is ' + str(x_train) + '\n')
    f.write('the test x array is ' + str(x_test) + '\n')
x_train = torch.FloatTensor(x_train)
x_test = torch.FloatTensor(x_test)
y_train, y_test = random_split(source_label, [train_size, test_size])
with open(desfilepath,'w') as f:
    f.write('the train y array is ' + str(y_train) + '\n')
    f.write('the test y array is ' + str(y_test))
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)

# 定义GetLoader类，继承Dataset方法，并重写__getitem__()和__len__()方法
class GetLoader(torch.utils.data.Dataset):
	# 初始化函数，得到数据
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label
    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels
    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)

# 通过GetLoader将数据进行加载，返回Dataset对象，包含data和labels
train_data = GetLoader(x_train, y_train)
test_data = GetLoader(x_test, y_test)

# 读取数据
train_datas = DataLoader(train_data, batch_size=4, shuffle=True, drop_last=False)
train_x = []
train_label = []
for x, y in train_datas:
    for i in range(len(x)):
        train_x.append(x[i])
        train_label.append(y[i])
train_x = torch.cat(train_x, dim=0)
train_x = train_x.reshape((train_size, Form1ng.xsize))
train_label = torch.cat(train_label, dim=0)
train_label = train_label.reshape((train_size, Form1ng.ysize))

test_datas = DataLoader(test_data, batch_size=4, shuffle=True, drop_last=False)
test_x = []
test_label = []
for x, y in test_datas:
    for i in range(len(x)):
        test_x.append(x[i])
        test_label.append(y[i])
test_x = torch.cat(test_x, dim=0)
test_x = test_x.reshape((test_size, Form1ng.xsize))
test_label = torch.cat(test_label, dim=0)
test_label = test_label.reshape((test_size, Form1ng.ysize))

from sklearn.preprocessing import StandardScaler
transform = StandardScaler()
train_x = transform.fit_transform(train_x)
test_x = transform.fit_transform(test_x)
train_x = torch.FloatTensor(train_x)
test_x = torch.FloatTensor(test_x)

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

maml = MamlModel(Form1ng.xsize, Form1ng.ysize, train_x, 8)
Y_train, Y_predict = maml(train_x, train_label)
for i in range(Form1ng.xsize + Form1ng.ysize):
    Y_train[i]
    Y_predict[i]

maml = MamlModel(Form1ng.xsize, Form1ng.ysize, train_x, 8)
optimer = optim.Adam(maml.parameters(), lr=0.01, weight_decay=1e-5)
loss_function = nn.MSELoss()
loss = 0

'''
下面：定义一些超参数，包括迭代次数，任务数量，对于原始参数更新的学习率(对于每一个任务更新的学习率，我直接定义在
了优化器里面)，原始的向量
'''
epoches = 4460
tasks = 9
beta = 0.0001

# 定义需要更新参数的矩阵
theta_matrix_w1 = torch.zeros(size=[9, 8, Form1ng.xsize])
theta_matrix_w1 = theta_matrix_w1.float()
theta_matrix_b1 = torch.zeros(size=[9, 1, 8])
theta_matrix_b1 = theta_matrix_b1.float()

theta_matrix_w2 = torch.zeros(size=[9, 8, 8])
theta_matrix_w2 = theta_matrix_w2.float()
theta_matrix_b2 = torch.zeros(size=[9, 1, 8])
theta_matrix_b2 = theta_matrix_b2.float()

theta_matrix_w3 = torch.zeros(size=[9, 8, 8])
theta_matrix_w3 = theta_matrix_w3.float()
theta_matrix_b3 = torch.zeros(size=[9, 1, 8])
theta_matrix_b3 = theta_matrix_b3.float()

theta_matrix_w4 = torch.zeros(size=[9, 8, 8])
theta_matrix_w4 = theta_matrix_w4.float()
theta_matrix_b4 = torch.zeros(size=[9, 1, 8])
theta_matrix_b4 = theta_matrix_b4.float()

theta_matrix_w5 = torch.zeros(size=[9, Form1ng.ysize, 8])
theta_matrix_w5 = theta_matrix_w5.float()
theta_matrix_b5 = torch.zeros(size=[9, 1, Form1ng.ysize])
theta_matrix_b5 = theta_matrix_b5.float()

ori_theta_w1 = torch.randn(size=[8, Form1ng.xsize])
ori_theta_w1 = ori_theta_w1.float()
ori_theta_b1 = torch.randn(size=[1, 8])
ori_theta_b1 = ori_theta_b1.float()

ori_theta_w2 = torch.randn(size=[8, 8])
ori_theta_w2 = ori_theta_w2.float()
ori_theta_b2 = torch.randn(size=[1, 8])
ori_theta_b2 = ori_theta_b2.float()

ori_theta_w3 = torch.randn(size=[8, 8])
ori_theta_w3 = ori_theta_w3.float()
ori_theta_b3 = torch.randn(size=[1, 8])
ori_theta_b3 = ori_theta_b3.float()

ori_theta_w4 = torch.randn(size=[8, 8])
ori_theta_w4 = ori_theta_w4.float()
ori_theta_b4 = torch.randn(size=[1, 8])
ori_theta_b4 = ori_theta_b4.float()

ori_theta_w5 = torch.randn(size=[Form1ng.ysize, 8])
ori_theta_w5 = ori_theta_w5.float()
ori_theta_b5 = torch.randn(size=[1, Form1ng.ysize])
ori_theta_b5 = ori_theta_b5.float()

meta_gradient_w1 = torch.zeros_like(ori_theta_w1)
meta_gradient_b1 = torch.zeros_like(ori_theta_b1)
meta_gradient_w2 = torch.zeros_like(ori_theta_w2)
meta_gradient_b2 = torch.zeros_like(ori_theta_b2)
meta_gradient_w3 = torch.zeros_like(ori_theta_w3)
meta_gradient_b3 = torch.zeros_like(ori_theta_b3)
meta_gradient_w4 = torch.zeros_like(ori_theta_w4)
meta_gradient_b4 = torch.zeros_like(ori_theta_b4)
meta_gradient_w5 = torch.zeros_like(ori_theta_w5)
meta_gradient_b5 = torch.zeros_like(ori_theta_b5)

#下面定义训练过程
def train(epoch):
    # 对每一个任务进行迭代(训练),保留每一个任务梯度下降之后的参数
    global ori_theta_w1, ori_theta_b1, ori_theta_w2, ori_theta_b2, ori_theta_w3, ori_theta_b3, ori_theta_b4, ori_theta_b5, ori_theta_w4, ori_theta_w5
    global theta_matrix_w1, theta_matrix_b1, theta_matrix_w2, theta_matrix_b2, theta_matrix_w3, theta_matrix_b3, theta_matrix_b4, theta_matrix_b5, theta_matrix_w4, theta_matrix_w5
    global meta_gradient_w1, meta_gradient_b1, meta_gradient_w2, meta_gradient_b2, meta_gradient_w3, meta_gradient_b3, meta_gradient_b4, meta_gradient_b5, meta_gradient_w4, meta_gradient_w5
    loss_sum = 0.0
    for i in range(tasks):
        maml.state_dict()['linear1.weight'].data = ori_theta_w1.data
        maml.state_dict()['linear1.bias'].data = ori_theta_b1.data
        maml.state_dict()['linear2.weight'].data = ori_theta_w2.data
        maml.state_dict()['linear2.bias'].data = ori_theta_b2.data
        maml.state_dict()['linear3.weight'].data = ori_theta_w3.data
        maml.state_dict()['linear3.bias'].data = ori_theta_b3.data
        maml.state_dict()['linear4.weight'].data = ori_theta_w4.data
        maml.state_dict()['linear4.bias'].data = ori_theta_b4.data
        maml.state_dict()['linear5.weight'].data = ori_theta_w5.data
        maml.state_dict()['linear5.bias'].data = ori_theta_b5.data
        optimer.zero_grad()
        Y_train, Y_predict = maml(train_x, train_label)
        loss_value = loss_function(Y_train, Y_predict)
        RMSE_loss_value = torch.sqrt(loss_value)
        loss_sum = loss_sum + RMSE_loss_value.data.item()
        RMSE_loss_value.backward()
        optimer.step()
        theta_matrix_w1[i, :] = maml.state_dict()['linear1.weight'].data
        theta_matrix_w2[i, :] = maml.state_dict()['linear2.weight'].data
        theta_matrix_w3[i, :] = maml.state_dict()['linear3.weight'].data
        theta_matrix_w4[i, :] = maml.state_dict()['linear4.weight'].data
        theta_matrix_w5[i, :] = maml.state_dict()['linear5.weight'].data

        theta_matrix_b1[i, :] = maml.state_dict()['linear1.bias'].data
        theta_matrix_b2[i, :] = maml.state_dict()['linear2.bias'].data
        theta_matrix_b3[i, :] = maml.state_dict()['linear3.bias'].data
        theta_matrix_b4[i, :] = maml.state_dict()['linear4.bias'].data
        theta_matrix_b5[i, :] = maml.state_dict()['linear5.bias'].data

    # 对每一个任务进行迭代（测试），利用保留的梯度下降之后的参数作为训练参数，计算梯度和
    for i in range(tasks):
        maml.state_dict()['linear1.weight'].data = theta_matrix_w1[i].data
        maml.state_dict()['linear2.weight'].data = theta_matrix_w2[i].data
        maml.state_dict()['linear3.weight'].data = theta_matrix_w3[i].data
        maml.state_dict()['linear4.weight'].data = theta_matrix_w4[i].data
        maml.state_dict()['linear5.weight'].data = theta_matrix_w5[i].data

        maml.state_dict()['linear1.bias'].data = theta_matrix_b1[i].data
        maml.state_dict()['linear2.bias'].data = theta_matrix_b2[i].data
        maml.state_dict()['linear3.bias'].data = theta_matrix_b3[i].data
        maml.state_dict()['linear4.bias'].data = theta_matrix_b4[i].data
        maml.state_dict()['linear5.bias'].data = theta_matrix_b5[i].data

        optimer.zero_grad()
        Y_test, Y_predict_test = maml(test_x, test_label)
        loss_value = loss_function(Y_test, Y_predict_test)
        RMSE_loss_value = torch.sqrt(loss_value)
        RMSE_loss_value.backward()
        optimer.step()
        meta_gradient_w1 = meta_gradient_w1 + maml.state_dict()['linear1.weight'].data
        meta_gradient_b1 = meta_gradient_b1 + maml.state_dict()['linear1.bias'].data
        meta_gradient_w2 = meta_gradient_w2 + maml.state_dict()['linear2.weight'].data
        meta_gradient_b2 = meta_gradient_b2 + maml.state_dict()['linear2.bias'].data
        meta_gradient_w3 = meta_gradient_w3 + maml.state_dict()['linear3.weight'].data
        meta_gradient_b3 = meta_gradient_b3 + maml.state_dict()['linear3.bias'].data
        meta_gradient_w4 = meta_gradient_w4 + maml.state_dict()['linear4.weight'].data
        meta_gradient_b4 = meta_gradient_b4 + maml.state_dict()['linear4.bias'].data
        meta_gradient_w5 = meta_gradient_w5 + maml.state_dict()['linear5.weight'].data
        meta_gradient_b5 = meta_gradient_b5 + maml.state_dict()['linear5.bias'].data
 # 更新初始的ori_theta
    ori_theta_w1 = ori_theta_w1 - beta * meta_gradient_w1 / tasks
    ori_theta_b1 = ori_theta_b1 - beta * meta_gradient_b1 / tasks
    ori_theta_w2 = ori_theta_w2 - beta * meta_gradient_w2 / tasks
    ori_theta_b2 = ori_theta_b2 - beta * meta_gradient_b2 / tasks
    ori_theta_w3 = ori_theta_w3 - beta * meta_gradient_w3 / tasks
    ori_theta_b3 = ori_theta_b3 - beta * meta_gradient_b3 / tasks
    ori_theta_w4 = ori_theta_w4 - beta * meta_gradient_w4 / tasks
    ori_theta_b4 = ori_theta_b4 - beta * meta_gradient_b4 / tasks
    ori_theta_w5 = ori_theta_w5 - beta * meta_gradient_w5 / tasks
    ori_theta_b5 = ori_theta_b5 - beta * meta_gradient_b5 / tasks
    loss = loss_sum / tasks
    Y_test, Y_predict = maml(test_x, test_label)
    ls = loss_function(Y_test, Y_predict)
    rlsv = torch.sqrt(loss_value)

for epoch in range(epoches):
    train(epoch)

torch.save(maml, ptfilepath)

with open('templates/train.html','r') as f:
    forming = f.read()
    position = re.search('</datalists>',forming)
    pos = position.start()
    forming = forming[:pos] + '<option>' + Form1ng.ptfilename + '</option>\n' + forming[pos + 17 + len(Form1ng.ptfilename):0]
os.remove('templates/train.html')
with open('templates/train.html','w') as f:
    f.write(forming)

with open('templates/predict.html','r') as f:
    forming = f.read()
    position = re.search('</datalists>',forming)
    pos = position.start()
    forming = forming[:pos] + '<option>' + Form1ng.ptfilename + '</option>\n' + forming[pos + 17 + len(Form1ng.ptfilename):0]
os.remove('templates/predict.html')
with open('templates/predict.html','w') as f:
    f.write(forming)