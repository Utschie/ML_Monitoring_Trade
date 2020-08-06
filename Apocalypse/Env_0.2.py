#专门写的一个环境类
#区别于第一代集成的环境，此环境在初始化时就把状态洗好了，或许可以节省读取状态的时间
import tensorflow as tf
from collections import deque
import numpy as np
import pandas as pd
import csv
import random
import re
from sklearn.decomposition import PCA
import os
import time
class Env():#定义一个环境用来与网络交互
    def __init__(self,filepath,result):
        self.result = result#获得赛果
        with open('D:\\data\\cidlist.csv') as f:
            reader = csv.reader(f)
            cidlist = [row[1] for row in reader]#得到cid对应表
        self.cidlist = list(map(float,cidlist))#把各个元素字符串类型转成浮点数类型
        self.filepath = filepath
        self.episode = self.episode_generator(self.filepath)#通过load_toNet函数得到当场比赛的episode生成器对象
        self.capital = 500#每场比赛有500欧可支配资金
        self.gesamt_revenue = 0#初始化实际收益
        self.invested = [[(0.,0.),(0.,0.),(0.,0.)]]#已投入资本，每个元素记录着一次投资的赔率和投入，分别对应胜平负三种赛果的投入，这里不用np.array因为麻烦
        self.statematrix = np.zeros((410,9))#初始化状态矩阵
        #传入原始数据，为一个不定长张量对象
        print('环境初始化完成')
     
    def episode_generator(self,filepath):#传入单场比赛文件路径，得到一个每一幕的generator
        data = pd.read_csv(filepath)#读取文件
        frametimelist=data.frametime.value_counts().sort_index(ascending=False).index#将frametime的值读取成列表
        for i in frametimelist:#其中frametimelist里的数据是整型
            state = data.groupby('frametime').get_group(i)#从第一次变盘开始得到当次转移
            state = np.array(state)#转成numpy多维数组
            #在填充成矩阵之前需要知道所有数据中到底有多少个cid
            statematrix=np.zeros((410,9))##statematrix应该是一个（1,410,8）的张量,元素为一个生成410*9的0矩阵（后来由于降维则不用这么用了）
            for j in state:
                cid = j[1]#得到浮点数类型的cid
                index = self.cidlist.index(cid)
                statematrix[index] = j#把对应矩阵那一行给它
            statematrix=np.delete(statematrix, 1, axis=-1)#去掉cid后，最后得到一个1*410*8的张量，这里axis是2或者-1（代表最后一个）都行
            self.statematrix = statematrix#把状态矩阵保存在环境对象里，用来算收益
            self.frametime = i
            yield self.statematrix,self.frametime

    def revenue(self,action):#收益计算器，根据行动和终止与否，计算收益给出，每次算一次revenue，capital都会变化，除了终盘
        #先把行动存起来
        max_host = self.statematrix[tf.argmax(self.statematrix)[2].numpy()][2]
        max_fair = self.statematrix[tf.argmax(self.statematrix)[3].numpy()][3]
        max_guest = self.statematrix[tf.argmax(self.statematrix)[4].numpy()][4]
        peilv = [max_host,max_fair,max_guest]#得到最高赔率向量
        peilv_action = list(zip(peilv,action))
        if self.capital >= sum(action):#如果剩下的资本还够执行行动，则把此次交易计入
            self.invested.append(peilv_action)#把本次投资存入invested已投入资本
        #计算本次行动的收益
        if self.statematrix.max(0)[0] ==0:#如果当前的状态是终盘状态,则清算所有赢的钱
            if self.result.host > self.result.guest:#主胜
                revenue = sum(i[0][0]*i[0][1] for i in self.invested )
            elif self.result.host == self.result.guest:#平
                revenue = sum(i[1][0]*i[1][1] for i in self.invested )
            else:#主负
                revenue = sum(i[2][0]*i[2][1] for i in self.invested )
            self.gesamt_revenue =self.gesamt_revenue + revenue
        elif self.capital < sum(action):#如果没到终盘，且action的总投资比所剩资本还多，则给revenue一个很大的负值给神经网络，但是对capital不操作，实际资本也不更改
            revenue = -200#则收益是个很大的负值（正常来讲revenue最大-50）
        else:
            revenue = -sum(action)
            self.capital += revenue#该局游戏的capital随着操作减少
            self.gesamt_revenue = self.gesamt_revenue + revenue
        return revenue
       
    def get_state(self):
        try:
            next_state,frametime=self.episode.__next__()
            done = False
        except:
            next_state = np.zeros((410,9))
            frametime = 0
            done = True
        return next_state, frametime,done,self.capital#网络从此取出下一幕
    
    def get_zinsen(self):
        zinsen  = self.gesamt_revenue/500.0
        return zinsen#这里必须是500.0，否则出来的是结果自动取整数部分，也就是0
    