#本模型是用LRCN的视频识别的方式做的分类模型，输出赛果的概率
#每一帧只有一个输入通道，为了能提取不同公司的重要性，直接用conv2D来读取
#CNN部分本版本使用GoogLeNet
#本版本使用的CNN让时序上所有帧都通过同一个网络
#先把batch里的每一帧都展开全部输入到googlenet里出个结果，再根据batch里的每个帧序列的长度把这所有的帧重新按原长度划分后放入Lstm中
#不过googlenet这么大的参数，4G显存的GPU是装不下的，仅仅batch_size=2,4G就装不下了
#而如果lstm的input_size = 1024,hidden_size个数为20000时，参数将有超过3.3亿个，显然无法接受
#lstm层，input_size=300,hidden_size=2000时，参数有3600万个，而输入改成50时，参数也有3200多万个
#lstm层，input_size=300,hidden_size=500时，参数320万，就还算可以接受，cuda也装得下————20201217
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
        #frametimelist = np.array(frametimelist),暂时不考虑位置参数
        framelist = np.array(framelist)
        #vectensor = self.mrx2vec(framelist)#LRCN里取消了Mrx2vec的过程，直接传入单帧序列然后用CNN和LSTM处理
        return torch.from_numpy(framelist)#传出一个帧列表,也可以把frametimelist一并传出来，此处暂不考虑位置参数的问题,是一个(seq_len,)


class Inception(nn.Module):#GoogLeNet的基础模块
    # c1 - c4为每条线路里的层的输出通道数
    def __init__(self, in_c, c1, c2, c3, c4):
        super().__init__()
        # 线路1，单1 x 1卷积层
        self.p1_1 = nn.Conv2d(in_c, c1, kernel_size=1)
        # 线路2，1 x 1卷积层后接3 x 3卷积层
        self.p2_1 = nn.Conv2d(in_c, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1 x 1卷积层后接5 x 5卷积层
        self.p3_1 = nn.Conv2d(in_c, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3 x 3最大池化层后接1 x 1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_c, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)  # 在通道维上连结输出   

class GlobalAvgPool2d(nn.Module):#copy自教程
    # 全局平均池化层可通过将池化窗口形状设置成输入的高和宽实现
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)

class GoogLeNet(nn.Module):#完全套用了教程中的GoogLeNet，
    def __init__(self):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                   nn.Conv2d(64, 192, kernel_size=3, padding=1),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                   Inception(256, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                   Inception(512, 160, (112, 224), (24, 64), 64),
                   Inception(512, 128, (128, 256), (24, 64), 64),
                   Inception(512, 112, (144, 288), (32, 64), 64),
                   Inception(528, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                   Inception(832, 384, (192, 384), (48, 128), 128),
                   GlobalAvgPool2d())

        self.net = nn.Sequential(self.b1, self.b2, self.b3, self.b4, self.b5, 
                    FlattenLayer())#这里去掉了原模型里的全连接层，因为要输入LSTM的cell
    
    def forward(self,x):
        return self.net(x)



class Lstm(nn.Module):#把CNN的结果输入LSTM里
    def __init__(self,max_seq_len):
        super().__init__()
        self.encoder = nn.LSTM(input_size=1024, 
                                hidden_size=max_seq_len,#这里应该是数据集中最长的序列长度，或者是做截断，选择离终赔最近的那几次变盘之类的，或者对帧采样固定长度
                                num_layers=1,#暂时就只有一层
                                bidirectional=True)
        self.decoder = nn.Linear(max_seq_len, 3)#把LSTM的输出

    def forward(self,inputs):
        return self.decoder(self.encoder(inputs))











def my_collate(batch):#由于默认下dataloader要求batch里的张量是相同尺寸，需要自己定义一个collate函数才能允许不同尺寸
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return data, target

def get_parameter_number(model):#参数统计
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == "__main__":
    root_path = 'D:\\data\\developing'
    dataset = BisaiDataset(root_path)
    print('数据集读取完成')
    loader = DataLoader(dataset, 32, shuffle=True,collate_fn = my_collate,num_workers=4)#num_workers>0情况下无法在交互模式下运行
    print('dataloader准备完成')
    conv_net = GoogLeNet().cuda()
    lstm_net = Lstm().cuda()
    print('网络构建完成')
    stat = get_parameter_number(conv_net)
    print(str(stat))
    lr, num_epochs = 0.001, 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    for epoch in range(1, num_epochs + 1):
        counter = 0
        start = time.time()
        for x, y in iter(loader):#x是一个存放data的列表，而data中每一个元素又是一个framelist和seq_len的二元组
            len_list = list(map(len,x))#把各个batch的长度取出来做一个列表
            conv_input = torch.cat(x,0).unsqueeze(1).cuda().float()#把所有batch拼接成一个大的放入卷积网络里，插入通道维，转成float()
            conv_output = conv_net(x)#得到第0维为batch_size的输出
            lstm_input = conv_output.split(len_list,0)#再按照各个batch的seq_len再划分开




            x = pad_sequence(x,batch_first=True).permute(0,2,1).float().cuda()#由于序列长度不同所以，再先填充最后两维再转置，使得x满足conv1d的输入要求
            #但是还需要使填充后的那些0不参与计算，所以可能需要制作掩码矩阵
            #或者需要时序全局最大池化层来消除填充的后果
            output = net(x).cpu()#x要求是一个固定shape的第0维是batch_size的张量，所以需要批量填充
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

     
