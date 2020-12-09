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
        self.filelist = [i+'\\'+k for i,j,k in os.walk(filepath) for k in k]#得到所有csv文件的路径列表
        self.lables = {'win':1,'lose':2,'draw':3}
    
    def __getitem__(self, index):
        # TODO
        i = False#用来标记函数是否成功的布尔变量
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        #这里需要注意的是，第一步：read one data，是一个dat
        while i == False:
            try:
                data_path = self.filelist[index]
                bisai_id = int(re.findall(r'\\(\d*?).csv',data_path)[0])
                # 2. Preprocess the data (e.g. torchvision.Transform).
                data = self.csv2frame(data_path)
                # 3. Return a data pair (e.g. image and label).
                lable = self.lablelist.loc[bisai_id].result
                lable = self.lables[lable]
                i = True#如果没错则i=True，跳出循环
            except Exception:
                self.filelist.remove(data_path)#如果出错，说明赛果里没有这场比赛，则在列表里去掉data_path
                continue#然后从try处再开始
        return data,lable      
       
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.filelist)

    def csv2frame(self,filepath):#给出单场比赛的csv文件路径，并转化成帧列表和对应变帧时间列表，以及比赛结果
        data = pd.read_csv(filepath)#读取文件
        data = data.drop(columns=['league','zhudui','kedui','companyname'])#去除非数字的列
        frametimelist=data.frametime.value_counts().sort_index(ascending=False).index#将frametime的值读取成列表
        framelist = np.zeros((len(frametimelist),601,10), dtype=float)#framelist为一个空列表,长度与frametimelist相同,一定要规定好具体形状和float类型，否则dataloader无法读取
        '''
        此处两个循环算法太慢，或许可以用merge并表的方式提高速度
        '''
        for i in frametimelist:
            state = data.groupby('frametime').get_group(i)#从第一次变盘开始得到当次转移
            state = np.array(state)#转成numpy多维数组
            #在填充成矩阵之前需要知道所有数据中到底有多少个cid
            statematrix=np.zeros((601,12),dtype=float)#因为cid_complete里共有306个cid；去掉非数字列后有12列
            for j in state:
                cid = j[1]#得到浮点数类型的cid
                index = cidlist.index(cid)
                statematrix[index] = j#把对应矩阵那一行给它
            statematrix=np.delete(statematrix,(0,1), axis=-1)#去掉frametime和cid列
            k = list(frametimelist).index(i)#找到i在frametimelist里的位置,由于frametimelist是ndarray，所以需要转成list取index 
            framelist[k] = statematrix#在framelist同样的位置中给元素赋值
        frametimelist = np.array(frametimelist)
        vectensor = self.mrx2vec(framelist)
        return vectensor#传出一个帧列表,也可以把frametimelist一并传出来，此处暂不考虑位置参数的问题
    
    def tsvd(self,frame):
        tsvd = TruncatedSVD(1)
        if frame.shape[0] != 1:
            newframe = tsvd.fit_transform(np.transpose(frame))#降维成（1,10）的矩阵
        else:
            pass
        return newframe
    
    def mrx2vec(self,framelist):#把截断奇异值的方法把矩阵变成向量(matrix2vec/img2vec)，传入：len(frametimelist)*(306*10),传出：len(frametimelist)*10
        veclist = np.array(list(map(self.tsvd,framelist))).squeeze()
        #veclist = veclist.transpose()
        vectensor = torch.from_numpy(veclist)#转成张量
        return vectensor#传出一个形状为(1,序列长度,10)的张量，因为后面传入模型之前，还需要做一下pad_sequence(0维是batch_size维)

    

class TextCNN(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        对于每一个样本，有以下思路：
        1.用mrx2vec把每一帧从307*10降到1*10，然后得到的len(frametimelist)*10做为一张单通道图片，
          再用二维卷积核(高为10，宽任意)把图片变成一个不定宽度的序列或图片，再用torch.nn.AdaptiveMaxPool1d/2d池化层转成相同长度序列
          最后输入MLP
        2.不使用mrx2vec,直接把307个公司视作307个通道(或者转置下把10个指标做10个通道)，用一维卷积核或满高度的二维卷积核把每一帧转成一个定长序列，然后再同上
        3.               
                        |---->mrx2vec-------->|                                                      
                        |                     |                                                      
          单场比赛------>              10*len(frametimelist)----->|Conv2D|---->n*len(frametimelist)---->|AdaptiveMaxPool1d/2d|--->定长序列---->|MLP|
                        |                     |                        |                                        ^                          
                        |---->Conv1D--------->|                        |--->1*len(frametimelist)----------------|
                                                                                                                |------------------------>|LSTM|
                                                                                                                                           
        ''' 
        #in_channels=10,意味着每个channel对应一个卷积核，然后这10个卷积核对10个通道过一遍后相加得到一个序列，叫做一个输出通道
        #out_channels=64,就是有64个独立的这样的操作
        #Conv1d/2d的第0维都是batch_size,输入的第0维也得是batch_size,
        #所以Conv2d输入形状是(batch_size,in_channels,img_height,img_width),Conv1d的输入形状是(batch_size,in_channels,seq_width)
        #....Conv2d输出.....(batch_size,out_channels,img_heights-kernel_heights+1,img_width-kernel_width+1),Conv1d输出(batch_size,out_channels,seq_width-kernel_width+1)
        #如果输入的形状不符，或者定义的in_channels和数据的in_channels不符则会出Error
        #卷积层的groups参数我也说不太清，但是会让参数减少，并且必须同时被输入和输出通道数整除
        #conv的卷积核是有bias偏置的，也就是说，即便所有元素为0，卷积输出也不为0，除非设置bias=False
        self.conv1 = nn.Conv1d(in_channels = 10, out_channels = 64, kernel_size = 3,bias = False,groups=2).float()#把（10*时序长度）的张量，把每一行当做单通道，通过核宽为2的一维卷积核转成（1*时序长度-2+1）的序列
        self.conv2 = nn.Conv1d(in_channels = 10, out_channels = 50, kernel_size = 5,bias = False,groups=2).float()#把核宽换成4
        #self.conv3 = nn.Conv2d(in_channels = 1, out_channels = 10, kernel_size = (4,4)).double()#
        self.pool1 = nn.AdaptiveMaxPool1d(1)#对每个通道输出的里输出一个最大值，需要用最大池化，来消除序列填充0的影响
        #一维池化层，用在conv1上，输出一个序列，池化层不改变通道数，如果conv层输入10个通道，则池化层也是过滤出10个通道
        #一维池化层的输入/输出形状是(batch_size,out_channels,width)
        self.pool2 = nn.AdaptiveMaxPool1d(1)#adaptiv的池化层无视序列长度均输出同一个
        #self.pool3 = nn.AdaptiveMaxPool2d((1,150)),二维池化层的输入/输出形状是(batch_size,out_channels,height,width)
        self.mlp = nn.Sequential(
            nn.Linear(64+50,120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 3)#输出节点需要为3，即输出三个值，另外不需要softmax层，因为使用nn.CrossEntropyLoss()时就已经用了softmax
        )

    


    
    def forward(self,inputs):
        output1 = self.pool1(F.relu(self.conv1(inputs)))
        output2 = self.pool2(F.relu(self.conv2(inputs)))
        #outputs3 = self.pool3(F.relu(self.conv3(embeddings.unsqueeze(0))))
        output3 = torch.cat([output1.squeeze(-1),output2.squeeze(-1)],1)#去掉最后一维后在channel维上合并，变成(batch_size,64+50)的张量，然后输入MLP
        output = self.mlp(output3)
        return output



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
    loader1 = DataLoader(dataset, 32, shuffle=True,collate_fn = my_collate,num_workers=1)#没法设定num_workers>0时无法在交互模式下使用，只能在命令行里跑
    loader2 = DataLoader(dataset, 32, shuffle=True,collate_fn = my_collate,num_workers=2)
    loader3 = DataLoader(dataset, 32, shuffle=True,collate_fn = my_collate,num_workers=4)
    loader4 = DataLoader(dataset, 32, shuffle=True,collate_fn = my_collate,num_workers=8)
    print('dataloader准备完成')
    train_iter_list = [iter(loader1),iter(loader2),iter(loader3),iter(loader4)]
    for i in train_iter_list:
        start = time.time()
        a = next(i)
        end = time.time()
        print('耗时'+str(end-start)+'秒')

    

   
    