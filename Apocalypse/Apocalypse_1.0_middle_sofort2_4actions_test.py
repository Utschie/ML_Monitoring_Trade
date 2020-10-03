#本模型只是比1.0_middle_sofort2_4actions的测试程序
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
        
class Q_Network(tf.keras.Model):
    def __init__(self,
                      n_companies=71,
                      n_actions=4):#有默认值的属性必须放在没默认值属性的后面
        self.n_companies = n_companies
        self.n_actions = n_actions
        super().__init__()#调用tf.keras.Model的类初始化方法
        self.dense1 = tf.keras.layers.Dense(units=144, activation=tf.nn.relu)#输入层
        self.dense2 = tf.keras.layers.Dense(units=144, activation=tf.nn.relu)#一个隐藏层
        self.dense3 = tf.keras.layers.Dense(units=144, activation=tf.nn.relu)
        self.dense4 = tf.keras.layers.Dense(units=144, activation=tf.nn.relu)
        self.dense5 = tf.keras.layers.Dense(units=144, activation=tf.nn.relu)
        self.dense6_v = tf.keras.layers.Dense(units=1, activation=tf.nn.relu)
        self.dense6_a = tf.keras.layers.Dense(units=self.n_actions)#输出层代表着在当前最大赔率前，买和不买的六种行动的价值

    def call(self,state): #输入从env那里获得的statematrix
        x = self.dense1(state)#输出神经网络
        x = self.dense2(x)#
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        v = self.dense6_v(x)
        a = self.dense6_a(x)
        q_values = v+(a-tf.reduce_mean(a, axis=1, keepdims=True))#Dueling DQN
        return q_values#q_value是一个（1,4）的张量

    def predict(self,state,capital):#用来对应动作
        q_values = self.call(state)#先根据jiangwei好的state求q值
        index = tf.squeeze(np.argwhere(np.sum(actions_table,axis=1)<=capital),axis=-1)#找出所有小于剩余资本的操作的索引
        new_q_value = []
        for i in index:
            new_q_value.append(q_values[0].numpy()[i])#将索引列表所对应的q值依次加入new_q_value里，从而形成了index和q值按顺序对应的两个列表
        action_index = index[tf.argmax(new_q_value)]#new_q_value的单个元素即index_list中对应位置的动作的q值
        return action_index#找出最大q值所对应的行动索引

    def filter(self,state,capital):
        q_values = self.call(state)
        index = np.argwhere(np.sum(actions_table,axis=1)<=capital)#找出所有小于剩余资本的操作的索引
        new_q_value = []
        for i in index:
            new_q_value.append(q_values[0].numpy()[i])#找出所有满足条件的操作的q值
        return new_q_value#返回所有满足条件的q值
    

def jiangwei(state,capital,mean_invested):#所有变量都归一化
    invested = [0.,0.,0.,0.,0.,0.]
    state=np.delete(state, 0, axis=-1)
    length = len(state)/410.0#出赔率的公司数归一化
    invested[0] = mean_invested[0]/25.0
    invested[1] = mean_invested[1]/500.0
    invested[2] = mean_invested[2]/25.0
    invested[3] = mean_invested[3]/500.0
    invested[4] = mean_invested[4]/25.0
    invested[5] = mean_invested[5]/500.0
    percenttilelist = [np.percentile(state,i,axis = 0)[1:4] for i in range(0,105,5)]
    percentile = np.vstack(percenttilelist)#把当前状态的0%-100%分位数放到一个矩阵里
    state = tf.concat((percentile.flatten()/25.0,[capital/500.0],invested,[length]),-1)#除以25是因为一般来讲赔率最高开到25
    state = tf.reshape(state,(1,71))#63个分位数数据+8个capital,frametime和mean_invested,length共72个输入
    return state


 
if __name__ == "__main__":
    start0 = time.time()
    summary_writer = tf.summary.create_file_writer('./tensorboard_1.0_middle_sofort2_4actions_test') #在代码所在文件夹同目录下创建tensorboard文件夹（本代码在jupyternotbook里跑，所以在jupyternotebook里可以看到）
    #########设置超参数
    resultlist = pd.read_csv('D:\\data\\results_20141130-20160630.csv',index_col = 0)#得到赛果和比赛ID的对应表
    actions_table = [[0,0,0],[5,0,0],[0,5,0],[0,0,5]]#给神经网络输出层对应一个行动表
    bisai_counter = 1
    target_Q = Q_Network()#初始化目标Q网络
    weights_path = 'D:\\data\\eval_Q_weights_1.0_middle_sofort2_4actions.ckpt'
    target_Q.load_weights(weights_path)
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
                state = jiangwei(state,capital,bianpan_env.mean_invested)#先降维，并整理形状，把capital放进去
                action_index = target_Q.predict(state,capital)#用predict，选择最优行动
                action = action_index#否则按着贪心选，这里[0]是因为predict返回的是一个单元素列表
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

