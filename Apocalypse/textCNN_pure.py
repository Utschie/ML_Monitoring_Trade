#本文件是纯粹用CNN，预处理是把每帧处理成相同大小，然后用CNN读取矩阵
#把单场比赛转成(1,10,1101)的图片，前面的奇异值分解再补上终赔
#4G显存刚刚好不可以承受128的batch，64的batch没问题————20201219
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
            frametimelist = [frametimelist[0]]+random.sample(list(frametimelist)[1:-1],498)+[frametimelist[-1]]#如果长度大于500,保留头尾，并在中间随机抽取498个，共计500个
            frametimelist.sort(reverse=True)#并降序排列
        for i in frametimelist:
            state = new_data[lables==i]#从第一次变盘开始得到当次转移
            #state = np.array(state)#不必转成numpy多维数组，因为已经是了
            state = np.delete(state,(0,1), axis=-1)#去掉frametime和cid
            #在填充成矩阵之前需要知道所有数据中到底有多少个cid
            framelist.append(state)
        statematrix=np.zeros((601,11),dtype=np.float64)#先建立一个空列表
        statematrix[:,0] = cidlist#把cidlist赋给第0列
        state = new_data[lables==frametimelist[-1]]#从第一次变盘开始得到当次转移
        state = np.array(state)#转成numpy多维数组
        state = np.delete(state,0, axis=-1)#只去掉frametime
        statelables = state[:,0]#得到state的cid
        statematrixlables = statematrix[:,0]#得到statematrix的cid
        def np_merge(lable):
            statematrix[statematrixlables==lable] = state[statelables == lable]
        vfunc = np.vectorize(np_merge)
        vfunc(statelables)
        statematrix=np.delete(statematrix,0, axis=-1)#再去掉cid列
        frametimelist = np.array(frametimelist)
        vectensor = self.mrx2vec(framelist)
        len_frame = vectensor.shape[0]
        if len_frame<500:
            vectensor = np.concatenate((np.zeros((500-len_frame,10),dtype=np.float64),vectensor),axis=0)#如果不足500，则在前面用0填充
        vectensor = np.concatenate((vectensor,statematrix),axis=0)
        vectensor = torch.from_numpy(vectensor.transpose(1,0))
        return vectensor.unsqueeze(0)#传出一个帧列表,也可以把frametimelist一并传出来，此处暂不考虑位置参数的问题
    
    def tsvd(self,frame):
        tsvd = TruncatedSVD(1)
        if frame.shape[0] != 1:
            newframe = tsvd.fit_transform(np.transpose(frame))#降维成（1,10）的矩阵
        else:
            return frame.reshape((10,1))#第一行需要reshape一下
        return newframe
    
    def mrx2vec(self,flist):#把截断奇异值的方法把矩阵变成向量(matrix2vec/img2vec)，传入：len(frametimelist)*(306*10),传出：len(frametimelist)*10
        vectensor = np.array(list(map(self.tsvd,flist))).squeeze()
        #veclist = veclist.transpose()
        #vectensor = torch.from_numpy(veclist)#转成张量
        return vectensor#传出一个形状为(1,序列长度,10)的张量，因为后面传入模型之前，还需要做一下pad_sequence(0维是batch_size维)




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
                    FlattenLayer(),nn.Linear(1024, 3))#这里去掉了原模型里的全连接层，因为要输入LSTM的cell
    
    def forward(self,x):
        return self.net(x)

def get_parameter_number(model):#参数统计
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == "__main__":
    root_path = 'D:\\data\\developing'
    dataset = BisaiDataset(root_path)
    print('数据集读取完成')
    loader = DataLoader(dataset, 128, shuffle=True,num_workers=4)#num_workers>0情况下无法在交互模式下运行
    print('dataloader准备完成')
    net = GoogLeNet().double().cuda()#双精度
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
            #但是还需要使填充后的那些0不参与计算，所以可能需要制作掩码矩阵
            #或者需要时序全局最大池化层来消除填充的后果
            x = x.double().cuda()
            output = net(x)#x要求是一个固定shape的第0维是batch_size的张量，所以需要批量填充
            output = output.cpu()
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
        torch.cuda.empty_cache()#一个epoch后释放显存
