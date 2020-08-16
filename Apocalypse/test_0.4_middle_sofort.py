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
        for i in frametimelist:#其中frametimelist里的数据是整型
            state = data.groupby('frametime').get_group(i)#从第一次变盘开始得到当次转移
            statematrix = np.array(state)#转成numpy多维数组
            #在填充成矩阵之前需要知道所有数据中到底有多少个cid
            self.statematrix=np.delete(statematrix, 1, axis=-1)#去掉cid后，最后得到一个1*410*8的张量，这里axis是2或者-1（代表最后一个）都行
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
            revenue = -100
        if action ==[0,0,0]:
            self.no_action_counter+=1#计算无行动率
        #计算本次行动的收益
        return revenue
       
    def get_state(self):
        next_state,frametime=self.episode.__next__()
        done = False
        if int(frametime) ==0:
            done = True
        return next_state, frametime,done,self.capital#网络从此取出下一幕
    
    def get_zinsen(self):#本局利润
        self.gesamt_touzi =500.0-self.capital
        zinsen  = float(self.gesamt_revenue)/float(self.gesamt_touzi+0.000001)
        return zinsen#这里必须是500.0，否则出来的是结果自动取整数部分，也就是0
        

class Q_Network(tf.keras.Model):
    def __init__(self,
                      n_companies=72,
                      n_actions=1331):#有默认值的属性必须放在没默认值属性的后面
        self.n_companies = n_companies
        self.n_actions = n_actions
        super().__init__()#调用tf.keras.Model的类初始化方法
        self.dense1 = tf.keras.layers.Dense(units=144, activation=tf.nn.relu)#输入层
        self.dense2 = tf.keras.layers.Dense(units=144, activation=tf.nn.relu)#一个隐藏层
        self.dense3 = tf.keras.layers.Dense(units=144, activation=tf.nn.relu)
        self.dense6 = tf.keras.layers.Dense(units=self.n_actions)#输出层代表着在当前最大赔率前，买和不买的六种行动的价值

    def call(self,state): #输入从env那里获得的statematrix
        x = self.dense1(state)#输出神经网络
        x = self.dense2(x)#
        x = self.dense3(x)
        q_value = self.dense6(x)#
        return q_value#q_value是一个（1,1331）的张量

    def predict(self, state):#用来对应动作
        q_values = self(state)
        return tf.argmax(q_values,axis=-1)#tf.argmax函数是返回最大数值的下标，用来对应动作
    

def jiangwei(state,capital,mean_invested):
    frametime = state[0][0]/80000.0#frametime最多80000秒之前开赔
    state=np.delete(state, 0, axis=-1)
    length = len(state)#出赔率的公司数
    percenttilelist = [np.percentile(state,i,axis = 0)[1:4] for i in range(0,105,5)]
    percentile = np.vstack(percenttilelist)#把当前状态的0%-100%分位数放到一个矩阵里
    state = tf.concat((percentile.flatten()/25.0,[capital],[frametime],mean_invested,[length]),-1)#除以25是因为一般来讲赔率最高开到25
    state = tf.reshape(state,(1,72))#63个分位数数据+8个capital,frametime和mean_invested,length共72个输入
    return state


if __name__ == "__main__":
    start0 = time.time()
    summary_writer = tf.summary.create_file_writer('./tensorboard_0.4_middle_sofort_test') #在代码所在文件夹同目录下创建tensorboard文件夹（本代码在jupyternotbook里跑，所以在jupyternotebook里可以看到）
    #########设置超参数
    #final_epsilon = 0.01            # 探索终止时的探索率
    epsilon = 0.0 
    resultlist = pd.read_csv('D:\\data\\results_20141130-20160630.csv',index_col = 0)#得到赛果和比赛ID的对应表
    actions_table = [[a,b,c] for a in range(0,55,5) for b in range(0,55,5) for c in range(0,55,5)]#给神经网络输出层对应一个行动表
    bisai_counter = 1
    revenue_list = []#记录近20场比赛的收益率
    zonglirun = 0.0#总利润
    target_Q = Q_Network()#初始化目标Q网络
    weights_path = 'D:\\data\\eval_Q_weights_0.4_middle_sofort.ckpt'
    target_Q.load_weights(weights_path)
    filefolderlist = os.listdir('F:\\test')
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
            state,frametime,done,capital =  bianpan_env.get_state()#把第一个状态作为初始化状态
            while True:
                state = jiangwei(state,capital,bianpan_env.mean_invested)#先降维，并整理形状，把capital放进去
                action_index = target_Q.predict(state)[0]#获得行动q_value
                if random.random() < epsilon:#如果落在随机区域
                    action = random.choice(range(0,1331))#action是一个坐标
                else:
                    action = action_index#否则按着贪心选，这里[0]是因为predict返回的是一个单元素列表
                bianpan_env.revenue(actions_table[action])
                next_state,frametime,done,next_capital = bianpan_env.get_state()#获得下一个状态,终止状态的next_state为0矩阵
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
            print('比赛'+filepath+'已测试完成'+'\n'+'用时'+str(end-start)+'秒\n')
    end0 = time.time()
    print('test总共用了'+str(end0-start0)+'秒')
