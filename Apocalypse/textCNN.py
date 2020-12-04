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
import os
import torch
from torch import nn
import torch.utils.data as Data
import  torch.nn.functional as F
import sys
import pandas as pd
import numpy as np
import csv

with open('D:\\data\\cidlist_public.csv') as f:#从cidlist_complete文件里读出306个cid来
    reader = csv.reader(f)
    cidlist = [row[1] for row in reader]#得到cid对应表
cidlist = list(map(float,cidlist))#把各个元素字符串类型转成浮点数类型
class Data_loader(object):#数据预处理器，把每一场比赛的固定时间点之前的数据转化成张量序列
    def __init__(self,big_filepath,result,timepoint):
        self.filepath = big_filepath#这里的filepath是总的最大的那个filepath
        self.result = result
        self.batch_list = self.shuffler()

    def loader(self,filepath):#给出单场比赛的csv文件路径，并转化成帧列表和对应变帧时间列表
        data = pd.read_csv(filepath)#读取文件
        data = data.drop(columns=['league','zhudui','kedui','companyname'])#去除非数字的列
        frametimelist=data.frametime.value_counts().sort_index(ascending=False).index#将frametime的值读取成列表
        framelist = list()#framelist为一个空列表
        for i in frametimelist:#其中frametimelist里的数据是整型
            state = data.groupby('frametime').get_group(i)#从第一次变盘开始得到当次转移
            state = np.array(state)#转成numpy多维数组
            #在填充成矩阵之前需要知道所有数据中到底有多少个cid
            statematrix=np.zeros((410,12))#
            for j in state:
                cid = j[1]#得到浮点数类型的cid
                index = cidlist.index(cid)
                statematrix[index] = j#把对应矩阵那一行给它
            statematrix=np.delete(statematrix,(0,1), axis=-1)#去掉frametime和cid列
            framelist.append(statematrix)
        framelist = np.array(framelist)#转成numpy数组
        frametimelist = np.array(frametimelist)
        return framelist,frametimelist

    def shuffler(self):#在完成一个epoch的学习后，对数据进行shuffle重新分组，得到一个mini_batch的列表



        return batch_list#返回新洗好的分batch列表，其中每个元素是一个装有32个文件名的列表



    def feeder(self):#每次调用都把一个准备好的mini_batch传递给网络


        return mini_batch



