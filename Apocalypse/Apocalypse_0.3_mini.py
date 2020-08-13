#最重要的是改变了revenue的计算方式，改成根据比赛结果自动计算revenue，另外终盘不进行投资
#首先要做的，是要清除nan这种bug————20200811
#由于这里无效行动的收益为0，随着时间增加wrong_action_rate不会显著下降
#一个重要的问题是，换了revenue计算方法后，loss在上升，而且是某种截断式的上升，就是突然接近0，后有突然升到很高，后又突然接近0。
#但是再没出现过nan
#这里的gesamt_revenue没有算入成本，所以最后算restcapital是要-500
#即时收益gamma值为1效果不太好，所以在此设为0.5
#优化器选择Adam，初始学习率改为0.0001
#另外可以考虑一下错误行动的负收益，此处为-100，因为-500时loss降得太快了
#当-500时差不多到十万次转移可能就可以了，所以这次尝试快速结束随机策略，或者降到很低————202020813
'''
即时收益+终赔不参与投资+错误行动收益为0+非标准化+10万次转移转贪心+-100负收益+gamma(0.9)+Adam(0.01)
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"#这个是使在tensorflow-gpu环境下只使用cpu
import tensorflow as tf
from collections import deque
import numpy as np
import pandas as pd
import csv
import random
import re
from sklearn.decomposition import TruncatedSVD
import time
import sklearn
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
        

class Q_Network(tf.keras.Model):
    def __init__(self,
                      n_companies=18,
                      n_actions=1331):#有默认值的属性必须放在没默认值属性的后面
        self.n_companies = n_companies
        self.n_actions = n_actions
        super().__init__()#调用tf.keras.Model的类初始化方法
        self.dense1 = tf.keras.layers.Dense(units=32, activation=tf.nn.relu)#输入层
        self.dense2 = tf.keras.layers.Dense(units=32, activation=tf.nn.relu)#一个隐藏层
        self.dense3 = tf.keras.layers.Dense(units=32, activation=tf.nn.relu)
        self.dense4 = tf.keras.layers.Dense(units=32, activation=tf.nn.relu)
        self.dense5 = tf.keras.layers.Dense(units=32, activation=tf.nn.relu)
        self.dense6 = tf.keras.layers.Dense(units=self.n_actions)#输出层代表着在当前最大赔率前，买和不买的六种行动的价值

    def call(self,state): #输入从env那里获得的statematrix
        x = self.dense1(state)#输出神经网络
        x = self.dense2(x)#
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        q_value = self.dense6(x)#
        return q_value#q_value是一个（1,1331）的张量

    def predict(self, state):#用来对应动作
        q_values = self(state)
        return tf.argmax(q_values,axis=-1)#tf.argmax函数是返回最大数值的下标，用来对应动作
    


def jiangwei(state,capital,mean_invested):
    tsvd = TruncatedSVD(1)
    max_host = state[tf.argmax(state)[2].numpy()][2]
    max_fair = state[tf.argmax(state)[3].numpy()][3]
    max_guest = state[tf.argmax(state)[4].numpy()][4]
    max = [max_host,max_fair,max_guest]
    frametime = state[0][0]#取出frametime时间
    state=np.delete(state, 0, axis=-1)#把frametime去掉，则state变成了（410,7）的矩阵
    state = tsvd.fit_transform(np.transpose(state))#降维成（410,1）的矩阵
    state = tf.concat((state.flatten(),[capital],[frametime],mean_invested,max),-1)#把降好维的state和capital与frametime连在一起，此时是412长度的一维张量
    state = tf.reshape(state,(1,18))
    return state

 
if __name__ == "__main__":
    start0 = time.time()
    summary_writer = tf.summary.create_file_writer('./tensorboard_0.3_mini') #在代码所在文件夹同目录下创建tensorboard文件夹（本代码在jupyternotbook里跑，所以在jupyternotebook里可以看到）
    #########设置超参数
    learning_rate = 0.01#学习率
    opt = tf.keras.optimizers.Adam(learning_rate)#设定最优化方法
    gamma = 0.9
    epsilon = 1.            # 探索起始时的探索率
    #final_epsilon = 0.01            # 探索终止时的探索率
    batch_size = 500
    resultlist = pd.read_csv('D:\\data\\results_20141130-20160630.csv',index_col = 0)#得到赛果和比赛ID的对应表
    actions_table = [[a,b,c] for a in range(0,55,5) for b in range(0,55,5) for c in range(0,55,5)]#给神经网络输出层对应一个行动表
    step_counter = 0
    learn_step_counter = 0
    target_repalce_counter = 0 
    bisai_counter = 1
    memory_size = 10000
    replay_buffer = deque(maxlen=memory_size)#建立一个记忆回放区
    eval_Q = Q_Network()#初始化行动Q网络
    target_Q = Q_Network()#初始化目标Q网络
    weights_path = 'D:\\data\\eval_Q_weights_mini_0.3.ckpt'
    filefolderlist = os.listdir('F:\\cleaned_data_20141130-20160630')
    ################下面是单场比赛的流程



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
            state,frametime,done,capital =  bianpan_env.get_state()#把第一个状态作为初始化状态
            with summary_writer.as_default():
                tf.summary.scalar("Capital", capital,step = bisai_counter)
            while True:
                if (step_counter % 1000 ==0) and (epsilon>0):
                    epsilon = epsilon-0.01#也就是经过10万次转移epsilon降到0
                state = jiangwei(state,capital,bianpan_env.mean_invested)#先降维，并整理形状，把capital放进去
                action_index = eval_Q.predict(state)[0]#获得行动q_value
                if random.random() < epsilon:#如果落在随机区域
                    action = random.choice(range(0,1331))#action是一个坐标
                else:
                    action = action_index#否则按着贪心选，这里[0]是因为predict返回的是一个单元素列表
                revenue = bianpan_env.revenue(actions_table[action])#根据行动和是否终赔计算收益
                next_state,frametime,done,next_capital = bianpan_env.get_state()#获得下一个状态,终止状态的next_state为0矩阵
                if done:
                    replay_buffer.append((state, action, revenue,jiangwei(next_state,next_capital,bianpan_env.mean_invested),1))
                    with summary_writer.as_default():
                        tf.summary.scalar('Zinsen',bianpan_env.get_zinsen(),step = bisai_counter)
                        tf.summary.scalar('rest_capital',bianpan_env.gesamt_revenue+500,step = bisai_counter)
                        tf.summary.scalar('wrong_action_rate',bianpan_env.wrong_action_counter/bianpan_env.action_counter,step = bisai_counter)
                        tf.summary.scalar('investion_rate',bianpan_env.gesamt_touzi/500.0,step = bisai_counter)
                        break
                else:#如果没终盘
                    replay_buffer.append((state, action, revenue,jiangwei(next_state,next_capital,bianpan_env.mean_invested),0))
                #这里需要标识一下终止状态，钱花光了就终止了
                state = next_state
                capital = next_capital
                #下面是参数更新过程
                if (step_counter >1000) and (step_counter%10 == 0) :#1000步之后每转移10次进行一次eval_Q的学习
                    if step_counter >= batch_size:
                        batch_state, batch_action, batch_revenue, batch_next_state ,batch_done= zip(*random.sample(replay_buffer, batch_size))#zip(*...)解开分给别人的意思 
                    else:
                        batch_state, batch_action, batch_revenue, batch_next_state ,batch_done= zip(*random.sample(replay_buffer, step_counter))
                    #y_true = batch_revenue+tf.reduce_max(target_Q.predict(np.array(batch_next_state)),axis = 1)*(1-np.array(batch_done))#reduce_max来返回最大值，暂不考虑折现率gamma,
                        #tensorflow中张量相乘是对应行相乘，所以eval_Q(batch_state)有多少列，one_hot就得有多少列，如下
                    #y_pred = tf.reduce_sum(tf.squeeze(eval_Q(np.array(batch_state)))*tf.one_hot(np.array(batch_action),depth=1331,on_value=1.0, off_value=0.0),axis=1)#one_hot来生成对应位置为1的矩阵，depth是列数，reduce_sum(axis=1)来求各行和转成一维张量
                    #tf.squeeze是用来去掉张量里所有为1的维度
                    with tf.GradientTape() as tape:
                        loss =  tf.keras.losses.mean_squared_error(y_true = batch_revenue+gamma*tf.reduce_max(tf.squeeze(target_Q(np.array(batch_next_state))),axis = -1)*(1-np.array(batch_done))
                        ,y_pred =tf.reduce_sum(tf.squeeze(eval_Q(np.array(batch_state)))*tf.one_hot(np.array(batch_action),depth=1331,on_value=1.0, off_value=0.0),axis=1))#y_true和y_pred都是第0维为batch_size的张量
                    grads = tape.gradient(loss, eval_Q.variables)
                    with summary_writer.as_default():
                        tf.summary.scalar('loss',loss,step = learn_step_counter)
                    opt.apply_gradients(grads_and_vars=zip(grads, eval_Q.variables))
                    learn_step_counter+=1#每学习一次，学习步数+1
                    print('已学习'+str(learn_step_counter)+'次')
                    if (learn_step_counter % 300 == 0) and (learn_step_counter > 0):#每学习300次，target_Q网络参数进行一次变量替换
                        eval_Q.save_weights(weights_path, overwrite=True)#保存并覆盖之前的检查点，储存权重
                        target_Q.load_weights(weights_path)#读取eval_Q刚刚保存的权重
                        target_repalce_counter+=1
                        print('目标Q网络已更新'+str(target_repalce_counter)+'次')
                step_counter+=1#每转移一次，步数+1
            end=time.time()
            bisai_counter+=1
            print('比赛'+filepath+'已完成'+'\n'+'用时'+str(end-start)+'秒\n')
    end0 = time.time()
    print('20141130-20160630总共用了'+str(end0-start0)+'秒')

