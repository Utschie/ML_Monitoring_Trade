#最小收益增量（去掉计算收益率）+终赔不参与投资+错误行动收益为0
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
        self.statematrix = np.zeros((410,9))#初始化状态矩阵
        self.action_counter=0.0
        self.wrong_action_counter = 0.0
        self.mean_host = [0.0,0.0]#保存已买主胜的平均赔率和投入
        self.mean_fair = [0.0,0.0]#保存已买平局的平均赔率和投入
        self.mean_guest = [0.0,0.0]#保存已买客胜的平均赔率和投入
        self.mean_invested = self.mean_host+self.mean_fair+self.mean_guest
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
        if self.capital >= sum(action):#如果剩下的资本还够执行行动，则capital里扣除本次交易费用
            self.capital = self.capital-sum(action)#资金变少
            self.action_counter+=1
            bevor_cost = self.mean_host[1]+self.mean_fair[1]+self.mean_guest[1]#之前的投入
            bevor_host_reward = (np.prod(self.mean_host)-bevor_cost)#如果主胜之前的投入的收益率
            bevor_fair_reward = (np.prod(self.mean_fair)-bevor_cost)#如果平
            bevor_guest_reward = (np.prod(self.mean_guest)-bevor_cost)#如果客胜
            bevor_reward = min(bevor_host_reward,bevor_fair_reward,bevor_guest_reward)#取其中最小值最为之前的收益率
            host_middle = self.mean_host[1]+peilv_action[0][1]#即新的主胜投入
            self.mean_host = [(np.prod(self.mean_host)+np.prod(peilv_action[0]))/(host_middle+0.00000000001),host_middle]
            fair_middle = self.mean_fair[1]+peilv_action[1][1]
            self.mean_fair = [(np.prod(self.mean_fair)+np.prod(peilv_action[1]))/(fair_middle+0.00000000001),fair_middle]
            guest_middle = self.mean_guest[1]+peilv_action[2][1]
            self.mean_guest = [(np.prod(self.mean_guest)+np.prod(peilv_action[2]))/(guest_middle+0.00000000001),guest_middle]
            self.mean_invested = self.mean_host+self.mean_fair+self.mean_guest
            now_cost = host_middle+fair_middle+guest_middle#新的总投入
            now_host_reward = (np.prod(self.mean_host)-now_cost)
            now_fair_reward = (np.prod(self.mean_fair)-now_cost)#如果平
            now_guest_reward = (np.prod(self.mean_guest)-now_cost)#如果客胜
            now_reward = min(now_host_reward,now_fair_reward,now_guest_reward)#本次行动后的最小收益率
            revenue = now_reward-bevor_reward#本步的收益增量作为revenue返回
            if self.result.host > self.result.guest:
                reward = max_host*action[0]-sum(action)
            elif self.result.host == self.result.guest:
                reward = max_fair*action[1]-sum(action)
            else:
                reward = max_guest*action[2]-sum(action)
            self.gesamt_revenue+=reward#计算实际货币收入并保存起来
        else:#如果不够执行行动
            self.action_counter+=1
            self.wrong_action_counter+=1
            revenue = 0.0#由于没有行动，原收益并未改变
        #计算本次行动的收益
        return revenue
       
    def get_state(self):
        next_state,frametime=self.episode.__next__()
        done = False
        if int(frametime) ==0:
            done = True
        return next_state, frametime,done,self.capital#网络从此取出下一幕
    
    def get_zinsen(self):
        self.gesamt_touzi =500.0-self.capital
        zinsen  = float(self.gesamt_revenue)/float(self.gesamt_touzi+0.000001)
        return zinsen#这里必须是500.0，否则出来的是结果自动取整数部分，也就是0
        
