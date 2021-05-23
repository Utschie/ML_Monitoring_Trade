#本程序是计算各帧的均值和方差的程序
#因为不确定普通K均值聚类的fit函数可不可以用多次来按照文件来增加数据集
#所以分别试验sklearn的普通kmeans和minibatchkmeans————20210522
#首先先求出标准化需要的均值和方差
#这里求均值是算进所有的0项的，所以均值和方差的绝对值才会很小————20210522
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
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
import gc#内存垃圾释放
with open('/home/jsy/data/cidlist_complete.csv') as f:
    reader = csv.reader(f)
    cidlist = [row[1] for row in reader]#得到cid对应表
cidlist = list(map(float,cidlist))#把各个元素字符串类型转成浮点数类型
train_path = ['/media/jsy/WD/train_20170601-20180210']
filepath=train_path
lablelist = pd.read_csv('/home/jsy/data/lablelist.csv',index_col = 0)#比赛id及其对应赛果的列表
filelist0 = list()
for x in filepath:#把filepath这个列表中的所有目录下的路径组合成一个列表
    filelist0+=[i+'/'+k for i,j,k in os.walk(x) for k in k]#得到所有csv文件的路径列表
#self.filelist0 = [i+'/'+k for i,j,k in os.walk(filepath) for k in k]#得到所有csv文件的路径列表
filelist = [data_path for data_path in filelist0 if int(re.findall(r'/(\d*?).csv',data_path)[0]) in  lablelist.index]#只保留有赛果的文件路径

def csv2framematrix(filepath):#从文件中把各帧拉直然后排成矩阵
    data = pd.read_csv(filepath)#读取文件
    data = data.drop(columns=['league','zhudui','kedui','companyname'])#去除非数字的列
    frametimelist=data.frametime.value_counts().sort_index(ascending=False).index#将frametime的值读取成列表
    framematrix =list()#framelist为一个空列表,长度与frametimelist相同,一定要规定好具体形状和float类型，否则dataloader无法读取
    '''
    此处两个循环算法太慢，用pandas更慢，完全抛弃pandas后，数据处理速度从109秒降到了10秒，降到10秒后cpu利用率20%，再往上提也提不上去了，可能需要C++或C来写了
    '''
    new_data = np.array(data)
    lables = new_data[:,0]
    for i in frametimelist:
        state = new_data[lables==i]#从第一次变盘开始得到当次转移
        #state = np.array(state)#不必转成numpy多维数组，因为已经是了
        #state = np.delete(state,(0,1), axis=-1)#去掉frametime和cid
        #在填充成矩阵之前需要知道所有数据中到底有多少个cid
        statematrix=np.zeros((601,11),dtype=np.float64)#先建立一个空列表
        statematrix[:,0] = cidlist#把cidlist赋给第0列
        state = np.array(state)#转成numpy多维数组
        state = np.delete(state,0, axis=-1)#只去掉frametime
        statelables = state[:,0]#得到state的cid
        statematrixlables = statematrix[:,0]#得到statematrix的cid
        def np_merge(lable):
            statematrix[statematrixlables==lable] = state[statelables == lable]
        vfunc = np.vectorize(np_merge)
        vfunc(statelables)
        statematrix=np.delete(statematrix,0, axis=-1)#再去掉cid列
        statevector = statematrix.flatten()#拉直
        framematrix.append(statevector)
    framematrix = np.array(framematrix)
    return framematrix
    
#下面是以增量的方式求标准化需要用到的均值和标准差
old_mean = np.zeros(6010)#初始化均值
old_var = np.zeros(6010)#初始化方差
old_n = 0#初始化样本量
for i in tqdm(range(0,len(filelist))):
    if i == 0:
        framematrix=csv2framematrix(filelist[i])
        print(old_mean)
        print(old_var)
    elif i%30 == 0:#每集齐100个文件，怕太多了内存爆掉
        print('old_n = '+str(old_n))
        now_mean = np.mean(framematrix,axis=0)#按列求出当前framematrix的均值
        now_n = len(framematrix)#当前framematrix的样本量
        print('now_n = '+ str(now_n))
        new_n = old_n+now_n
        print('new_n = '+str(new_n))#输出新的均值
        new_mean = (old_n*old_mean+now_n*now_mean)/new_n#求出新的均值
        now_var = np.var(framematrix,axis=0)#按列求出当前framematrix的方差
        new_var = (old_n*(old_var+(new_mean-old_mean)**2)+now_n*(now_var+(new_mean-now_mean)**2))/new_n#计算新的方差
        old_mean = new_mean#用新的mean代替旧的mean
        old_var = new_var#用新的var代替旧的var
        old_n = new_n#用新的n代替旧的n
        print(old_mean)
        print(old_var)
        del framematrix
        gc.collect()#删除之前的framematrix，垃圾回收
        framematrix = csv2framematrix(filelist[i])#给framematrix重新赋值
    else:
        framematrix = np.vstack((framematrix,csv2framematrix(filelist[i])))#其他情况就是不断堆积framematrix
#最后再把剩下的不满足100个的framematrix加上      
now_mean = np.mean(framematrix,axis=0)#按列求出当前framematrix的均值
now_n = len(framematrix)#当前framematrix的样本量
print('now_n = '+ str(now_n))
new_n = old_n+now_n
print('new_n = '+str(new_n))#输出新的均值
new_mean = (old_n*old_mean+now_n*now_mean)/new_n#求出新的均值
now_var = np.var(framematrix,axis=0)#按列求出当前framematrix的方差
new_var = (old_n*(old_var+(new_mean-old_mean)**2)+now_n*(now_var+(new_mean-now_mean)**2))/new_n#计算新的方差
old_mean = new_mean#用新的mean代替旧的mean
old_var = new_var#用新的var代替旧的var
old_n = new_n#用新的n代替旧的n
np.save('/home/jsy/data/old_mean0.npy',old_mean)
np.save('/home/jsy/data/old_var0.npy',old_var)
np.save('/home/jsy/data/old_n0.npy',old_n)
print(old_n)
print(old_mean)
print(old_var)
del framematrix
gc.collect()#删除之前的framematrix，垃圾回收
        
        
    
