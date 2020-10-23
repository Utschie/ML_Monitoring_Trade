#本模型是不经过筛选行动，直接将错误行动reward为0的PPO模型
#本模型是nofilter3模型，与第2版的区别在于行动变成单位变成0.2————20201020
#此外，取消时间点的限制
#同时恢复每500步一学习的方式
#结果还是一样，也是解除随机后方差变得很大，然后critic的loss还变大了，就感觉不太好————20201023

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
        self.capital = 500.#每场比赛有500欧可支配资金
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
            revenue = 0.
        if action ==[0.,0.,0.]:
            revenue = 0.
            self.no_action_counter+=1#计算无行动率
        #计算本次行动的收益
        return revenue
       
    def get_state(self):
        next_state=self.episode.__next__()
        done = False
        if self.frametime ==0.0:
            done = True
        return next_state,self.frametime,done,self.capital#网络从此取出下一幕
    
    def get_zinsen(self):
        self.gesamt_touzi =500.0-self.capital
        zinsen  = float(self.gesamt_revenue)/float(self.gesamt_touzi+0.000001)
        return zinsen#这里必须是500.0，否则出来的是结果自动取整数部分，也就是0
 
class Critic_Network(tf.keras.Model):
    def __init__(self,n_actions=4):#有默认值的属性必须放在没默认值属性的后面
        self.n_actions = n_actions
        super().__init__()#调用tf.keras.Model的类初始化方法
        self.dense1 = tf.keras.layers.Dense(units=600, activation=tf.nn.relu)#输入层
        self.dense2 = tf.keras.layers.Dense(units=600, activation=tf.nn.relu)#一个隐藏层
        self.dense2_d = tf.keras.layers.Dropout(0.5)
        self.dense3 = tf.keras.layers.Dense(units=600, activation=tf.nn.relu)
        self.dense3_d = tf.keras.layers.Dropout(0.5)
        self.dense4 = tf.keras.layers.Dense(units=600, activation=tf.nn.relu)
        self.dense4_d = tf.keras.layers.Dropout(0.5)
        self.dense5 = tf.keras.layers.Dense(units=600, activation=tf.nn.relu)
        self.dense5_d = tf.keras.layers.Dropout(0.5)
        self.dense6_v = tf.keras.layers.Dense(units=1)
     

    def call(self,state): #输入从env那里获得的statematrix
        x = self.dense1(state)#输出神经网络
        x = self.dense2(x)#
        x = self.dense2_d(x)
        x = self.dense3(x)
        x = self.dense3_d(x)
        x = self.dense4(x)
        x = self.dense4_d(x)
        x = self.dense5(x)
        x = self.dense5_d(x)
        v = self.dense6_v(x)
        return v#v是当前的状态价值函数值


class Critic(object):
    def __init__(self):
        self.net = Critic_Network()
        self.lr = 0.0001
        self.opt = tf.keras.optimizers.Adam(self.lr,amsgrad=True)#设定最优化方法
    
    def learn(self,batch_state,batch_discounted_r):
        for i in range(3):#重复学10次
            with tf.GradientTape() as tape:
                batch_v = self.net(batch_state)#求出这一场比赛所有转移的
                advantage = batch_discounted_r - batch_v
                loss = tf.reduce_mean(tf.square(advantage))
            grads = tape.gradient(loss, self.net.variables)
            self.opt.apply_gradients(grads_and_vars=zip(grads, self.net.variables))#更新参数
        return loss
        
    def get_advantage(self,batch_state,batch_discounted_r):
        batch_v = self.net(batch_state)
        advantage = batch_discounted_r - batch_v
        return advantage


class Actor_Network(tf.keras.Model):#给actor定义的policy网络
    def __init__(self,n_actions=4):#有默认值的属性必须放在没默认值属性的后面
        self.n_actions = n_actions
        super().__init__()#调用tf.keras.Model的类初始化方法
        self.dense1 = tf.keras.layers.Dense(units=600, activation=tf.nn.relu)#输入层
        self.dense2 = tf.keras.layers.Dense(units=600, activation=tf.nn.relu)#一个隐藏层
        self.dense2_d = tf.keras.layers.Dropout(0.5)
        self.dense3 = tf.keras.layers.Dense(units=600, activation=tf.nn.relu)
        self.dense3_d = tf.keras.layers.Dropout(0.5)
        self.dense4 = tf.keras.layers.Dense(units=600, activation=tf.nn.relu)
        self.dense4_d = tf.keras.layers.Dropout(0.5)
        self.dense5 = tf.keras.layers.Dense(units=600, activation=tf.nn.relu)
        self.dense5_d = tf.keras.layers.Dropout(0.5)
        self.dense6 = tf.keras.layers.Dense(units=self.n_actions)#输出层代表着在当前最大赔率前，买和不买的六种行动的价值


    def call(self,state): #输入从env那里获得的statematrix
        x = self.dense1(state)#输出神经网络
        x = self.dense2(x)#
        x = self.dense2_d(x)
        x = self.dense3(x)
        x = self.dense3_d(x)
        x = self.dense4(x)
        x = self.dense4_d(x)
        x = self.dense5(x)
        x = self.dense5_d(x)
        parameters = self.dense6(x)#Dueling DQN
        return parameters#parameters是一个（1,4）的张量，是决定随机策略分布的参数向量

    def possibility(self,state):#用来对应动作
        parameters = self.call(state)#先根据jiangwei好的state求参数向量
        possibilities = tf.nn.softmax(parameters)#把这些动作改成概率
        return possibilities#返回可选行动及其概率

class Actor(object):
    def __init__(self):
        self.net = Actor_Network()
        self.old_net = Actor_Network()
        self.lr = 0.0001
        self.opt = tf.keras.optimizers.Adam(self.lr,amsgrad=True)#设定最优化方法
        self.clip_epsilon = 0.2
    
    def choose_action(self,state):
        possibilities = self.net.possibility(state)#获得行动概率二元组并解耦
        index = np.random.choice(range(4), p=np.array(possibilities).ravel())#根据概率选择索引
        return index

    def learn(self,batch_state,batch_capital,batch_action,batch_discounted_r):
        self.update_old()#更新旧网络参数
        for i in range(3):#重复10次
            with tf.GradientTape() as tape:
                one_hot_matrix = tf.one_hot(np.array(batch_action),depth=4,on_value=1.0, off_value=0.0)
                batch_parameters = self.net(tf.squeeze(batch_state))#获得parameters的值  
                pi = tf.nn.softmax(batch_parameters)#把parameters们都softmax化成概率
                pi_prob = tf.reduce_sum(pi*one_hot_matrix,axis=1)
                batch_parameters = self.old_net(tf.squeeze(batch_state))#获得parameters的值  
                old_pi = tf.nn.softmax(batch_parameters)#把parameters们都softmax化成概率
                old_pi_prob = tf.reduce_sum(old_pi*one_hot_matrix,axis=1)
                advantage = critic.get_advantage(batch_state,batch_discounted_r)
                ratio = pi_prob/(old_pi_prob+1e-8)
                surr = ratio*advantage
                aloss = -tf.reduce_mean(tf.minimum(surr,tf.clip_by_value(ratio, 1.-self.clip_epsilon, 1.+self.clip_epsilon)*advantage))
            grads = tape.gradient(aloss, self.net.variables)
            self.opt.apply_gradients(grads_and_vars=zip(grads, self.net.variables))#更新参数
        return aloss
    
    def update_old(self):
        for old_param, param in zip(self.old_net.trainable_weights, self.net.trainable_weights):
            old_param.assign(param)
        
            



class Memory(object):#这个memory是没达到一个batch或者到盘末就清空
    def __init__(self):
        self.memory = deque()#建立储存区
    def store(self,transition):#把每次的转移传给它，这个转移里只包含state,capital和action
        self.memory.append(transition)
    def get_memory(self):
        return self.memory#把记忆还给它
    def clear(self):
        self.memory = deque()
  

            
def jiangwei(state,capital,frametime,mean_invested):#所有变量都归一化
    invested = [0.,0.,0.,0.,0.,0.]
    state=np.delete(state, 0, axis=-1)
    frametime = frametime/50000.0
    length = len(state)/410.0#出赔率的公司数归一化
    invested[0] = mean_invested[0]/25.0
    invested[1] = mean_invested[1]/500.0
    invested[2] = mean_invested[2]/25.0
    invested[3] = mean_invested[3]/500.0
    invested[4] = mean_invested[4]/25.0
    invested[5] = mean_invested[5]/500.0
    percenttilelist = [np.percentile(state,i,axis = 0)[1:4] for i in range(0,105,5)]
    percentile = np.vstack(percenttilelist)#把当前状态的0%-100%分位数放到一个矩阵里
    state = tf.concat((percentile.flatten()/25.0,[capital/500.0],[frametime],invested,[length]),-1)#除以25是因为一般来讲赔率最高开到25
    state = tf.reshape(state,(1,72))#63个分位数数据+8个capital,frametime和mean_invested,length共72个输入
    return state



if __name__ == "__main__":
    summary_writer = tf.summary.create_file_writer('./tensorboard_1.0_middle_PPO_nofilter3') #在代码所在文件夹同目录下创建tensorboard文件夹（本代码在jupyternotbook里跑，所以在jupyternotebook里可以看到）
    summary_writer2 = tf.summary.create_file_writer('./tensorboard_1.0_middle_PPO_nofilter3/use_out_time')
    summary_writer3 = tf.summary.create_file_writer('./tensorboard_1.0_middle_PPO_nofilter3/max_frametime')
    summary_writer4 = tf.summary.create_file_writer('./tensorboard_1.0_middle_PPO_nofilter3/used_steps')
    summary_writer5 = tf.summary.create_file_writer('./tensorboard_1.0_middle_PPO_nofilter3/bisai_steps')
    summary_writer6 = tf.summary.create_file_writer('./tensorboard_1.0_middle_PPO_nofilter3/actor_loss')
    summary_writer7 = tf.summary.create_file_writer('./tensorboard_1.0_middle_PPO_nofilter3/critic_loss')
    start0 = time.time()
    epsilon = 1.            # 探索起始时的探索率
    #final_epsilon = 0.01            # 探索终止时的探索率
    gamma = 0.99999#平均每场比赛2000步来算的话，0.999差不多了
    resultlist = pd.read_csv('D:\\data\\results_20141130-20160630.csv',index_col = 0)#得到赛果和比赛ID的对应表
    actions_table = [[0.,0.,0.],[0.2,0.,0.],[0.,0.2,0.],[0.,0.,0.2]]#给神经网络输出层对应一个行动表
    step_counter = 0
    learn_step_counter = 0 
    step_in_critic = 0
    bisai_counter = 1
    critic_weights_path = 'D:\\data\\critic_weights_1.0_middle_PPO_nofilter3.ckpt'
    actor_weights_path = 'D:\\data\\actor_weights_1.0_middle_PPO_nofilter3.ckpt'
    filefolderlist = os.listdir('F:\\cleaned_data_20141130-20160630')
    actor = Actor()#实例化一个actor
    actor.net.save_weights(actor_weights_path, overwrite=True)#保存网络参数
    critic = Critic()#实例化一个critic
    memory = Memory()#初始化比赛记忆

    #下面是挨场比赛训练
    for i in filefolderlist:#挨个文件夹训练
        filelist = os.listdir('F:\\cleaned_data_20141130-20160630\\'+i)
        for j in filelist:#挨场比赛训练
            start=time.time()
            filepath = 'F:\\cleaned_data_20141130-20160630\\'+i+'\\'+j#文件路径
            bisai_id = int(re.findall(r'\\(\d*?).csv',filepath)[0])#从filepath中得到bisai代码的整型数
            try:
                result = resultlist.loc[bisai_id]#其中result.host即为主队进球，result.guest则为客队进球
            except Exception:#因为有的比赛结果没有存进去
                continue
            bianpan_env = Env(filepath,result)#每场比赛做一个环境
            memory.clear()#每场比赛开始前要清空记忆
            state,frametime,done,capital =  bianpan_env.get_state()#把第一个状态作为初始化状态
            max_frametime = bianpan_env.max_frametime#得到本场比赛最大的frametime
            end_switch = False
            bisai_steps = 0
            used_steps = 0
            while True:
                if step_counter >= 2000000:
                    epsilon = 0.0#如果超过200万次转移，转贪心 
                state = jiangwei(state,capital,frametime,bianpan_env.mean_invested)#先降维，并整理形状，把capital放进去
                if epsilon == 1.:#如果落在随机区域
                    action  = random.randint(0,3)
                else:
                    action = actor.choose_action(state)
                revenue = bianpan_env.revenue(actions_table[action])#计算收益
                next_state,next_frametime,done,next_capital = bianpan_env.get_state()#获得下一个状态,终止状态的next_state为0矩阵
                bisai_steps+=1
                if end_switch == False:#如果没花光
                    use_out_time = 1#没花光就当做1
                    used_steps+=1
                if (next_capital<= 0) and (end_switch == False):
                    use_out_time = frametime
                    end_switch = True
                if done:#终盘时储存信息，同时更新actor，清除actor内存
                    learn_step_counter+=1
                    transition = np.array((state,capital,action,revenue))#先把当下的存起来
                    memory.store(transition)
                    with summary_writer.as_default():
                        tf.summary.scalar('Zinsen',bianpan_env.get_zinsen(),step = bisai_counter)
                        tf.summary.scalar('rest_capital',bianpan_env.gesamt_revenue+500,step = bisai_counter)
                        tf.summary.scalar('investion_rate',bianpan_env.gesamt_touzi/500.0,step = bisai_counter)
                    with summary_writer2.as_default():
                        tf.summary.scalar('times',use_out_time,step =bisai_counter)
                    with summary_writer3.as_default():
                        tf.summary.scalar('times',bianpan_env.max_frametime,step =bisai_counter)
                    with summary_writer4.as_default():
                        tf.summary.scalar('steps',used_steps,step =bisai_counter)
                    with summary_writer5.as_default():
                        tf.summary.scalar('steps',bisai_steps,step =bisai_counter)
                    v = critic.net(jiangwei(next_state,next_capital,next_frametime,bianpan_env.mean_invested))#得到下一状态的状态价值
                    batch_memory = memory.get_memory()
                    batch_state,batch_capital,batch_action,batch_revenue = zip(*batch_memory)#把memory解开
                    if len(batch_state) == 1:#如果刚好batch_state里只有一次转移，那么直接跳出不学了
                        break
                    batch_discounted_r = []
                    for r in batch_revenue[::-1]:#折现
                        v = r + gamma * v
                        batch_discounted_r.append(v)
                    batch_discounted_r.reverse()
                    actor_loss = actor.learn(np.array(batch_state),np.array(batch_capital),np.array(batch_action),np.array(batch_discounted_r))
                    critic_loss = critic.learn(np.array(batch_state),np.array(batch_discounted_r))   
                    memory.clear()#清空memory            
                    with summary_writer6.as_default():
                        tf.summary.scalar('losses',actor_loss,step = learn_step_counter)     
                    with summary_writer7.as_default():
                        tf.summary.scalar('losses',critic_loss,step = learn_step_counter)          
                    break
                elif bisai_steps % 500 ==0:
                    learn_step_counter+=1
                    transition = np.array((state,capital,action,revenue))
                    memory.store(transition)
                    v = critic.net(jiangwei(next_state,next_capital,next_frametime,bianpan_env.mean_invested))#得到终盘的状态价值
                    batch_memory = memory.get_memory()
                    batch_state,batch_capital,batch_action,batch_revenue = zip(*batch_memory)#把memory解开
                    batch_discounted_r = []
                    for r in batch_revenue[::-1]:
                        v = r + gamma * v
                        batch_discounted_r.append(v)
                    batch_discounted_r.reverse()
                    actor_loss = actor.learn(np.array(batch_state),np.array(batch_capital),np.array(batch_action),np.array(batch_discounted_r))
                    critic_loss = critic.learn(np.array(batch_state),np.array(batch_discounted_r))   
                    memory.clear()#清空memory
                    with summary_writer6.as_default():
                        tf.summary.scalar('losses',actor_loss,step = learn_step_counter)     
                    with summary_writer7.as_default():
                        tf.summary.scalar('losses',critic_loss,step = learn_step_counter) 
                    state = next_state
                    capital = next_capital
                    frametime = next_frametime
                else:
                    transition = np.array((state,capital,action,revenue))
                    memory.store(transition)
                    state = next_state
                    capital = next_capital
                    frametime = next_frametime
                step_counter+=1#每转移一次，步数+1
            end=time.time()
            bisai_counter+=1
            print('比赛'+filepath+'已完成'+'\n'+'用时'+str(end-start)+'秒\n')
    end0 = time.time()
    print('20141130-20160630总共用了'+str(end0-start0)+'秒')





