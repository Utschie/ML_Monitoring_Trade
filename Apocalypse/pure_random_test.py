#本程序是纯随机的测试程序
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"#这个是使在tensorflow-gpu环境下只使用cpu
import tensorflow as tf
from collections import deque
import numpy as np
import pandas as pd
import csv
import random
import re
import time
import sklearn
import math
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
        self.action_counter=0.0
        self.no_action_counter = 0.0
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
        self.max_frametime = frametimelist[0]
        for i in frametimelist:#其中frametimelist里的数据是整型
            state = data.groupby('frametime').get_group(i)#从第一次变盘开始得到当次转移
            statematrix = np.array(state)#转成numpy多维数组
            #在填充成矩阵之前需要知道所有数据中到底有多少个cid
            self.statematrix=np.delete(statematrix, 1, axis=-1)#去掉cid后，最后得到一个1*410*8的张量，这里axis是2或者-1（代表最后一个）都行
            self.frametime = i
            yield self.statematrix

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
            host_middle = self.mean_host[1]+peilv_action[0][1]#即新的主胜投入
            self.mean_host = [(np.prod(self.mean_host)+np.prod(peilv_action[0]))/(host_middle+0.00000000001),host_middle]
            fair_middle = self.mean_fair[1]+peilv_action[1][1]
            self.mean_fair = [(np.prod(self.mean_fair)+np.prod(peilv_action[1]))/(fair_middle+0.00000000001),fair_middle]
            guest_middle = self.mean_guest[1]+peilv_action[2][1]
            self.mean_guest = [(np.prod(self.mean_guest)+np.prod(peilv_action[2]))/(guest_middle+0.00000000001),guest_middle]
            self.mean_invested = self.mean_host+self.mean_fair+self.mean_guest
            if self.result.host > self.result.guest:
                revenue = max_host*action[0]-sum(action)
            elif self.result.host == self.result.guest:
                revenue = max_fair*action[1]-sum(action)
            else:
                revenue = max_guest*action[2]-sum(action)
            self.gesamt_revenue+=revenue#最终计算收益时在加上，以表示所有赢得钱，因为后面要除以总投资
        else:#如果不够执行行动
            self.action_counter+=1
            self.wrong_action_counter+=1
            revenue = -200
        if action ==[0,0,0]:
            revenue = 0
            self.no_action_counter+=1#计算无行动率
        #计算本次行动的收益
        return revenue
       
    def get_state(self):
        next_state=self.episode.__next__()
        done = False
        if self.frametime ==0.0:
            done = True
        return next_state,done,self.capital#网络从此取出下一幕
    
    def get_zinsen(self):
        self.gesamt_touzi =500.0-self.capital
        zinsen  = float(self.gesamt_revenue)/float(self.gesamt_touzi+0.000001)
        return zinsen#这里必须是500.0，否则出来的是结果自动取整数部分，也就是0
        

 
if __name__ == "__main__":
    start0 = time.time()
    summary_writer = tf.summary.create_file_writer('./tensorboard_1.0_middle_sofort2_4actions_test/pure_random') #在代码所在文件夹同目录下创建tensorboard文件夹（本代码在jupyternotbook里跑，所以在jupyternotebook里可以看到）
    #########设置超参数
    resultlist = pd.read_csv('D:\\data\\results_20141130-20160630.csv',index_col = 0)#得到赛果和比赛ID的对应表
    actions_table = [[0,0,0],[5,0,0],[0,5,0],[0,0,5]]#给神经网络输出层对应一个行动表
    bisai_counter = 1
    filefolderlist = os.listdir('F:\\test')
    zonglirun = 0.0#总利润
    revenue_list = []#记录近20场比赛的收益率
    ################下面是单场比赛的流程
    for i in filefolderlist:#挨个文件夹训练
        filelist = os.listdir('F:\\test\\'+i)
        for j in filelist:#挨场比赛训练
            start=time.time()
            filepath = 'F:\\test\\'+i+'\\'+j#文件路径
            bisai_id = int(re.findall(r'\\(\d*?).csv',filepath)[0])#从filepath中得到bisai代码的整型数
            try:
                result = resultlist.loc[bisai_id]#其中result.host即为主队进球，result.guest则为客队进球
            except Exception:#因为有的比赛结果没有存进去
                continue
            bianpan_env = Env(filepath,result)#每场比赛做一个环境
            state,done,capital =  bianpan_env.get_state()#把第一个状态作为初始化状态 
            while True:
                qualified_index = tf.squeeze(np.argwhere(np.sum(actions_table,axis=1)<=capital),axis=-1)#找到符合条件的行动的index_list
                action = random.choice(qualified_index)#在qualified_index中随机选取一个动作,注意随机挑选出返回的是列表
                bianpan_env.revenue(actions_table[action])#根据行动和是否终赔计算收益
                next_state,done,next_capital = bianpan_env.get_state()#获得下一个状态,终止状态的next_state为0矩阵
                if done:
                    with summary_writer.as_default():
                        revenue_list.append(bianpan_env.get_zinsen())
                        if len(revenue_list)==20:
                            tf.summary.scalar('20_mean_Zinsen',np.mean(revenue_list),step = bisai_counter)#每20场算一下平均利率
                            revenue_list=[]#清空revenue_list
                        tf.summary.scalar('Zinsen',bianpan_env.get_zinsen(),step = bisai_counter)
                        tf.summary.scalar('rest_capital',bianpan_env.gesamt_revenue+500,step = bisai_counter)
                        tf.summary.scalar('wrong_action_rate',bianpan_env.wrong_action_counter/bianpan_env.action_counter,step = bisai_counter)
                        tf.summary.scalar('investion_rate',bianpan_env.gesamt_touzi/500.0,step = bisai_counter)
                        tf.summary.scalar('no_action_rate',bianpan_env.no_action_counter/bianpan_env.action_counter,step = bisai_counter)
                        zonglirun+=bianpan_env.gesamt_revenue
                        tf.summary.scalar('总利润',zonglirun,step=bisai_counter)#看总曲线是否增长
                        break
                state = next_state
                capital = next_capital
            end=time.time()
            bisai_counter+=1
            print('比赛'+filepath+'已完成'+'\n'+'用时'+str(end-start)+'秒\n')
    end0 = time.time()
    print('test总共用了'+str(end0-start0)+'秒')
