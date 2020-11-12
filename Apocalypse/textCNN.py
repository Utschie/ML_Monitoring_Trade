#本模型是利用textCNN来做的一个特征提取模型，用的anaconda的tensorflow环境下的pytorch1.5
#不确定是直接把状态作为conv的输入还是先经过一层embedding再进行conv
#不过如果需要embedding的话那可能需要graphembedding
#其实由于原始数据是一帧一帧dataframe连在一起，然后长短不一，所以其实可以看做一个视频分类的问题—————20201112
#可能会用到conv3D以及CRNN
#中间空白的帧（即有的时候不变盘）或许可以用之前最近的那一次变盘填充，作为当前帧————20201112
import os
import torch
from torch import nn
import torch.utils.data as Data
import  torch.nn.functional as F
import sys
import pandas as pd
import numpy as np


class Data_loader(object):#数据预处理器，把每一场比赛的固定时间点之前的数据转化成张量序列
    def __init__(self,big_filepath,result,timepoint):
        self.filepath = big_filepath#这里的filepath是总的最大的那个filepath
        self.result = result
        self.batch_list = self.shuffler()

    def episode_generator(self,filepath):#这里的filepath是大filepath下单场比赛的filepath
        data = pd.read_csv(filepath)#读取文件
        frametimelist=data.frametime.value_counts().sort_index(ascending=False).index#将frametime的值读取成列表
        self.max_frametime = frametimelist[0]
        for i in frametimelist:#其中frametimelist里的数据是整型
            state = data.groupby('frametime').get_group(i)#从第一次变盘开始得到当次转移
            statematrix = np.array(state)#转成numpy多维数组
            #在填充成矩阵之前需要知道所有数据中到底有多少个cid
            self.statematrix=np.delete(statematrix, 1, axis=-1)#去掉cid后，最后得到一个1*410*8的张量，这里axis是2或者-1（代表最后一个）都行
            self.frametime = i
            yield self.statematrix

    def shuffler(self):#在完成一个epoch的学习后，对数据进行shuffle重新分组，得到一个mini_batch的列表



        return batch_list#返回新洗好的分batch列表，其中每个元素是一个装有32个文件名的列表



    def feeder(self):#每次调用都把一个准备好的mini_batch传递给网络


        return mini_batch



