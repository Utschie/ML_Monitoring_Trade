#0.本文件假定奇异值分解是视频分类中的光流，那么单帧矩阵则作为10通道的一维图像流或者单通道二维图像流，如此采用双流法
#
#1.或者其实可以分别以比赛为单位训练lstm，用cnn网络以帧为单位训练矩阵（共约1亿个矩阵）然后全连接层用2-3层，最后一层是3个单元
#  然后两个网络都训练好后，再把CNN和LSTM训练好的模型的倒数第二层，比如各有200个单元，然后合并后仅仅训练合并后的那层到输出
#
#2.再或者就是把训练集分成两部分，因为矩阵有1亿多个，而参数最多也就几百万，完全可以拿一小部分数据part1来先单帧预训练CNN
#  然后把训练好的CNN当做特征提取器，在part2上训练LSTM
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
from prefetch_generator import BackgroundGenerator
from torch.utils.tensorboard import SummaryWriter
with open('C:\\data\\cidlist_complete.csv') as f:
    reader = csv.reader(f)
    cidlist = [row[1] for row in reader]#得到cid对应表
cidlist = list(map(float,cidlist))#把各个元素字符串类型转成浮点数类型
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
class BisaiDataset(Dataset):#数据预处理器
    def __init__(self,filepath):
        self.lablelist = pd.read_csv('C:\\data\\lablelist.csv',index_col = 0)#比赛id及其对应赛果的列表
        self.filelist0 = [i+'\\'+k for i,j,k in os.walk(filepath) for k in k]#得到所有csv文件的路径列表
        self.filelist = [data_path for data_path in self.filelist0 if int(re.findall(r'\\(\d*?).csv',data_path)[0]) in  self.lablelist.index]#只保留有赛果的文件路径
        self.lables = {'win':0,'draw':1,'lose':2}#分类问题要从0开始编号,而且要对应好了表中的顺序编，
    
    def __getitem__(self, index):
        #todo
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
        if len(frametimelist)>250:
            frametimelist = [frametimelist[0]]+random.sample(list(frametimelist)[1:-1],248)+[frametimelist[-1]]#如果长度大于500,保留头尾，并在中间随机抽取498个，共计500个
            frametimelist.sort(reverse=True)#并降序排列
        for i in frametimelist:
            state = new_data[lables==i]#从第一次变盘开始得到当次转移
            #state = np.array(state)#不必转成numpy多维数组，因为已经是了
            state = np.delete(state,(0,1), axis=-1)#去掉frametime和cid
            #在填充成矩阵之前需要知道所有数据中到底有多少个cid
            framelist.append(state)
        frametimelist = np.array(frametimelist)
        vectensor = self.mrx2vec(framelist)
        len_frame = vectensor.shape[0]
        if len_frame<250:
            vectensor = np.concatenate((np.zeros((250-len_frame,10),dtype=np.float64),vectensor),axis=0)#如果不足500，则在前面用0填充
        vectensor = torch.from_numpy(vectensor)
        return vectensor#传出一个帧列表,也可以把frametimelist一并传出来，此处暂不考虑位置参数的问题
    
    def tsvd(self,frame):
        tsvd = TruncatedSVD(1)
        if frame.shape[0] != 1:
            newframe = tsvd.fit_transform(np.transpose(frame))#降维成（1,10）的矩阵
        else:
            return frame.reshape((10,1))#第一行需要reshape一下
        return newframe
    
    def mrx2vec(self,flist):#把截断奇异值的方法把矩阵变成向量(matrix2vec/img2vec)，传入：len(frametimelist)*(306*10),传出：len(frametimelist)*10
        vectensor = np.array(list(map(self.tsvd,flist))).squeeze(2)
        #veclist = veclist.transpose()
        #vectensor = torch.from_numpy(veclist)#转成张量
        return vectensor#传出一个形状为(1,序列长度,10)的张量，因为后面传入模型之前，还需要做一下pad_sequence(0维是batch_size维)





