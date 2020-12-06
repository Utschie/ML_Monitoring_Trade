#本模型是利用textCNN来做的一个特征提取模型，用的anaconda的tensorflow环境下的pytorch1.5
#不确定是直接把状态作为conv的输入还是先经过一层embedding再进行conv
#不过如果需要embedding的话那可能需要graphembedding
#其实由于原始数据是一帧一帧dataframe连在一起，然后长短不一，所以其实可以看做一个视频分类的问题—————20201112
#可能会用到conv3D以及CRNN
#中间空白的帧（即有的时候不变盘）或许可以用之前最近的那一次变盘填充，作为当前帧————20201112
#Data_loader里准备数据部分里的循环部分或许可以用map方法提高一下效率，否则一个样本就几万次循环太慢了————20201127
#20160701-20190224总共有66709场比赛，20141130-20160630共有37897场比赛，总共104606场比赛————20201203
#本程序用的是20141130-20160630的37897场比赛做训练集，用20160701-20190224做验证集和测试集————20201203
#最新的cidlist共有600个cid，其中有一些奇怪的3000开头的cid，还没想好要不要去掉————20201204
#cidlist_complete是全部的cid文件，cid_publice是前后半段时间都共有的cid共有306个————20201204
#暂时用public的公共cid，因为如果用全部cid可能需要打乱全部10万个训练集的次序，就很麻烦，倒不如用大家都有的一直活着的公司数据————20201204
#本程序的开发暂时使用D盘data文件夹下的developing中的数据，即几天的数据，用于开发时使用————20201204
#可能需要给developing文件夹下的文件专门做一个lablelist————20201205
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

with open('D:\\data\\cidlist_public.csv') as f:
    reader = csv.reader(f)
    cidlist = [row[1] for row in reader]#得到cid对应表
cidlist = list(map(float,cidlist))#把各个元素字符串类型转成浮点数类型
class BisaiDataset(Dataset):#数据预处理器
    def __init__(self,filepath):
        self.lablelist = pd.read_csv('D:\\data\\lablelist_developing.csv',index_col = 0)#比赛id及其对应赛果的列表
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
        framelist = list(range(len(frametimelist)))#framelist为一个空列表,长度与frametimelist相同
        def data2framelist(i):
            state = data.groupby('frametime').get_group(i)#从第一次变盘开始得到当次转移
            state = np.array(state)#转成numpy多维数组
            #在填充成矩阵之前需要知道所有数据中到底有多少个cid
            statematrix0=np.zeros((306,12))#因为cid_public里共有306个cid；去掉非数字列后有12列
            def state2matrix(j):
                cid = j[1]#得到浮点数类型的cid
                index = cidlist.index(cid)
                statematrix0[index] = j#把对应矩阵那一行给它
            statematrix = np.fromiter(map(state2matrix,state),dtype = np.float64)#用np.fromiter实现从map中
            statematrix=np.delete(statematrix,(0,1), axis=-1)#去掉frametime和cid列
            k = list(frametimelist).index(i)#找到i在frametimelist里的位置,由于frametimelist是ndarray，所以需要转成list取index 
            framelist[k] = statematrix#在framelist同样的位置中给元素赋值
        framelist = map(data2framelist,frametimelist)
        framelist = np.fromiter(framelist,dtype = np.float64)#让map函数运行并转成numpy数组
        frametimelist = np.array(frametimelist)
        return (framelist,frametimelist)#传出一个单帧和对应位置的元组,以及拥有三个值的分类变量result
    

class TextCNN(nn.Module):
    def __init__(self,kernel_sizes,):
        super().__init__()
        self.pool = GlobalMaxPool1d()
        self.convs = nn.ModuleList() 
    

    def mrx2vec(self,framelist):#把截断奇异值的方法把矩阵变成向量(matrix2vec/img2vec)，传入：len(frametimelist)*(306*10),传出：len(frametimelist)*10
        veclist = np.fromiter(map(self.tsvd,framelist),dtype = np.float64) 
        return veclist

    def tsvd(self,frame):
        tsvd = TruncatedSVD(1)
        if frame.shape[0] != 1:
            newframe = tsvd.fit_transform(np.transpose(frame))#降维成（1,7）的矩阵
        else:
            pass
        return newframe

    
    def forward(self, inputs):
        embeddings = self.mrx2vec(inputs)





    



