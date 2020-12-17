#本版本是LRCN的brief版
#他把每一帧的状态矩阵都理解成拥有10个通道的1维序列，每个长度是601
#这个10通道的1维序列通过CNN（因为没有时间联系）提取特征后，再输入到LSTM里
#这里保留了CNN最后的全连接层，并把输出数改为300，googlenet改成1d后，参数变成370多万————20201217
#lstm需要把hidden_size降到500，也就是说，在预处理数据时，对序列长度>500的序列应该进行帧抽取————20201217
#本模型暂时采用保留头尾的纯均匀采样，之后如果可能再采取tsn模型的段共识函数的方法————20201217
#其实还有一种可能，就是完全弃用RNN，只使用CNN进行做分类模型，然后再根据每场比赛已经走过的帧加权投票————20201217
#预处理数据是把所有数据都变成500长度，不足的用0矩阵在序列前填充————20201217
#经过目前的改造，总模型参数423万，其中conv模型142万，lstm模型281万————20201217
#再度精简的convnet里有86万个参数————20201217
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import sys
import pandas as pd
import numpy as np
import csv
import random
import re
from sklearn.decomposition import TruncatedSVD
from torch.nn.utils.rnn import pad_sequence#用来填充序列
import time

with open('D:\\data\\cidlist_complete.csv') as f:
    reader = csv.reader(f)
    cidlist = [row[1] for row in reader]#得到cid对应表
cidlist = list(map(float,cidlist))#把各个元素字符串类型转成浮点数类型
class BisaiDataset(Dataset):#数据预处理器
    def __init__(self,filepath):
        self.lablelist = pd.read_csv('D:\\data\\lablelist.csv',index_col = 0)#比赛id及其对应赛果的列表
        self.filelist0 = [i+'\\'+k for i,j,k in os.walk(filepath) for k in k]#得到所有csv文件的路径列表
        self.filelist = [data_path for data_path in self.filelist0 if int(re.findall(r'\\(\d*?).csv',data_path)[0]) in  self.lablelist.index]#只保留有赛果的文件路径
        self.lables = {'win':0,'lose':1,'draw':2}#分类问题要从0开始编号，否则出错
    
    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        #这里需要注意的是，第一步：read one data，是一个dat
        data_path = self.filelist[index]
        bisai_id = int(re.findall(r'\\(\d*?).csv',data_path)[0])
        # 2. Preprocess the data (e.g. torchvision.Transform).
        data = self.csv2frame(data_path)
        # 3. Return a data pair (e.g. image and label).
        lable = self.lablelist.loc[bisai_id].result
        lable = self.lables[lable]
        return data,lable      
       
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.filelist)


    def csv2frame(self,filepath):#给出单场比赛的csv文件路径，并转化成帧列表和对应变帧时间列表，以及比赛结果
        data = pd.read_csv(filepath)#读取文件
        data = data.drop(columns=['league','zhudui','kedui','companyname'])#去除非数字的列
        frametimelist=data.frametime.value_counts().sort_index(ascending=False).index#将frametime的值读取成列表
        framelist =list()#framelist为一个空列表,长度与frametimelist相同,一定要规定好具体形状和float类型，否则dataloader无法读取
        '''
        此处两个循环算法太慢，用pandas更慢，完全抛弃pandas后，数据处理速度从109秒降到了10秒，降到10秒后cpu利用率20%，再往上提也提不上去了，可能需要C++或C来写了
        '''
        new_data = np.array(data)
        lables = new_data[:,0]
        if len(frametimelist)>500:
            frametimelist = random.sample(list(frametimelist),500)#如果长度大于500，随机抽取500个
            frametimelist.sort(reverse=True)#并降序排列
        for i in frametimelist:
            statematrix=np.zeros((601,11),dtype=float)#先建立一个空列表
            statematrix[:,0] = cidlist#把cidlist赋给第0列
            state = new_data[lables==i]#从第一次变盘开始得到当次转移
            state = np.array(state)#转成numpy多维数组
            state = np.delete(state,0, axis=-1)#只去掉frametime
            statelables = state[:,0]#得到state的cid
            statematrixlables = statematrix[:,0]#得到statematrix的cid
            def np_merge(lable):
                statematrix[statematrixlables==lable] = state[statelables == lable]
            vfunc = np.vectorize(np_merge)
            vfunc(statelables)
            statematrix=np.delete(statematrix,0, axis=-1)#再去掉cid列
            framelist.append(statematrix)
        frametimelist = np.array(frametimelist)
        framelist = np.array(framelist)
        len_frame = framelist.shape[0]
        if len_frame<500:
            framelist = np.concatenate((np.zeros((500-len_frame,601,10),dtype=float),framelist),axis=0)#如果不足500，则在前面用0填充
        #vectensor = self.mrx2vec(framelist)#LRCN里取消了Mrx2vec的过程，直接传入单帧序列然后用CNN和LSTM处理
        return framelist.transpose(0,2,1)#传出一个帧列表,也可以把frametimelist一并传出来，此处暂不考虑位置参数的问题,是一个(seq_len,)

class Inception(nn.Module):#把conv2d改成了1d，然后改一下参数
    # c1 - c4为每条线路里的层的输出通道数
    def __init__(self, in_c=10, c1=24, c2=(6,24), c3=(6,24), c4 = 24):#输入通道10，输出通道统一为24
        super().__init__()
        # 线路1，单1 x 1卷积层
        self.p1_1 = nn.Conv1d(in_c, c1, kernel_size=1)#输出通道6
        # 线路2，1 x 1卷积层后接3 x 3卷积层
        self.p2_1 = nn.Conv1d(in_c, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv1d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1 x 1卷积层后接5 x 5卷积层
        self.p3_1 = nn.Conv1d(in_c, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv1d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3 x 3最大池化层后接1 x 1卷积层
        self.p4_1 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv1d(in_c, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)  # 在通道维上连结输出,默认是24*4=96个通道 


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)

class ConvNet(nn.Module):#把2d改成了1d，把输入通道1改成了10，然后取消前面的conv，直接输入inception
    def __init__(self):
        super().__init__()
        self.b4 = nn.Sequential(Inception(),
                   Inception(96, 128, (128, 192), (32, 96), 64),
                   Inception(480, 128, (128, 192), (24, 64), 64),
                   #Inception(448, 128, (144, 192), (32, 64), 64),#暂时去掉一层以减少参数
                   Inception(448, 144, (32, 64), (32, 112), 128),
                   nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        self.b5 = nn.Sequential(#Inception(448, 64, (64, 128), (32, 128), 128),#暂时去掉一层以减少参数
                   Inception(448, 64, (96,128), (32, 192), 128),
                   nn.AdaptiveAvgPool1d(1))

        self.net = nn.Sequential(self.b4, self.b5, 
                    FlattenLayer(),nn.Linear(512, 200))#这里把原模型的输出改成了200，然后输入到lSTM层
    
    def forward(self,x):
        return self.net(x)



class Lstm(nn.Module):#把CNN的结果输入LSTM里
    def __init__(self):
        super().__init__()
        self.encoder = nn.LSTM(input_size=200, 
                                hidden_size=500,#选择对帧进行保留首尾的均匀截断采样
                                num_layers=1,#暂时就只有一层
                                bidirectional=True)
        self.decoder = nn.Linear(1000, 3)#把LSTM的输出

    def forward(self,inputs):
        output, _= self.encoder(inputs.permute(1,0,2))#inputs需要转置一下再输入lstm层，因为pytorch要求第0维为长度，第二维才是batch_size
        return self.decoder(output[-1])#把最后一个时间步的输出输入MLP

class ConvLstm(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_net = ConvNet()
        self.lstm_net = Lstm()
    def forward(self,x):
        conv_input = x.reshape((32*500,10,601)).cuda().float()#把所有batch拼接成一个大的放入卷积网络里，插入通道维，转成float()
        conv_output = self.conv_net(conv_input)#得到第0维为batch_size的输出
        lstm_input = conv_output.reshape((32,500,200))#再按batch_size拆开
        lstm_output = self.lstm_net(lstm_input)
        return lstm_output



def get_parameter_number(model):#参数统计
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == "__main__":
    root_path = 'D:\\data\\developing'
    dataset = BisaiDataset(root_path)
    print('数据集读取完成')
    loader = DataLoader(dataset, 32, shuffle=True,num_workers=4)#num_workers>0情况下无法在交互模式下运行
    print('dataloader准备完成')
    net = ConvLstm().cuda()
    print('网络构建完成')
    stat1 = get_parameter_number(net)
    print(str(stat1))
    lr, num_epochs = 0.001, 5
    optimizer= torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    for epoch in range(1, num_epochs + 1):
        counter = 0
        start = time.time()
        for x, y in iter(loader):
            '''
            len_list = list(map(len,x))#把各个batch的长度取出来做一个列表
            conv_input = torch.cat(x,0).unsqueeze(1).cuda().float()#把所有batch拼接成一个大的放入卷积网络里，插入通道维，转成float()
            conv_output = conv_net(x)#得到第0维为batch_size的输出
            lstm_input = conv_output.split(len_list,0)#再按照各个batch的seq_len再划分开
            '''
            #但是还需要使填充后的那些0不参与计算，所以可能需要制作掩码矩阵
            #或者需要时序全局最大池化层来消除填充的后果
            x = x.cuda().float()
            output = net(x).cuda()#x要求是一个固定shape的第0维是batch_size的张量，所以需要批量填充
            l = loss(output, y)
            optimizer.zero_grad() # 梯度清零，等价于net.zero_grad()
            l.backward()
            optimizer.step()
            end = time.time()
            train_period = end-start
            counter+=1
            print('第'+str(epoch)+'个epoch已学习'+str(counter)+'个batch,'+'用时'+str(train_period)+'秒')
            print('本batch的赛果为'+str(y))
            print('filelist长度为'+str(len(dataset.filelist)))
            start = time.time()
        print('epoch %d, loss: %f' % (epoch, l.item()))

     
