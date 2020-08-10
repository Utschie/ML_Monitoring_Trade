#用截断奇异值分解法降维可以把网络参数降到20000以下，这样可以用更大的batch_size,有更快的速度以及更大的学习率————20200810
#不过batch_size还没有改，需要试一下————20200810
#batch_size改成500和350万个参数下32大小的batch_size相同都是1秒5次转移，学习率提高了一个数量级暂时用0.001，然后随机episilong提高一个数量级，即随机10万次，跑一天，看看情况————20200810
#还是要增加一个能在tensorboard查看损失函数的语句————20200810
#损失函数出现nan，据说是梯度爆炸的原因————20200810
#但是我怀疑也可能是隐藏层单元太少，现在调成每层300也才70多万个参数————20200810
#又爆炸了，这次尝试降维后标准化————20200810
#但我怀疑也有可能是新添加的变量出了问题
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
        self.invested = [[(0.,0.),(0.,0.),(0.,0.)]]#已投入资本，每个元素记录着一次投资的赔率和投入，分别对应胜平负三种赛果的投入，这里不用np.array因为麻烦
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
        if self.capital >= sum(action):#如果剩下的资本还够执行行动，则把此次交易计入累计投资，并更新平均赔率
            self.invested.append(peilv_action)#把本次投资存入invested已投入资本
            self.action_counter+=1
            host_middle = self.mean_host[1]+peilv_action[0][1]#即新的主胜投入
            self.mean_host = [(np.prod(self.mean_host)+np.prod(peilv_action[0]))/host_middle,host_middle]
            fair_middle = self.mean_fair[1]+peilv_action[1][1]
            self.mean_fair = [(np.prod(self.mean_fair)+np.prod(peilv_action[1]))/fair_middle,fair_middle]
            guest_middle = self.mean_guest[1]+peilv_action[2][1]
            self.mean_guest = [(np.prod(self.mean_guest)+np.prod(peilv_action[2]))/guest_middle,guest_middle]
            self.mean_invested = self.mean_host+self.mean_fair+self.mean_guest
        else:
            self.action_counter+=1
            self.wrong_action_counter+=1
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
            next_state = np.zeros((410,8))
            frametime = 0
            done = True
        return next_state, frametime,done,self.capital#网络从此取出下一幕
    
    def get_zinsen(self):
        self.gesamt_touzi = np.sum(np.sum(self.invested,axis=1),axis=0)[1]
        zinsen  = float(self.gesamt_revenue)/float(self.gesamt_touzi+0.000001)
        return zinsen#这里必须是500.0，否则出来的是结果自动取整数部分，也就是0
        

class Q_Network(tf.keras.Model):
    def __init__(self,
                      n_companies=18,
                      n_actions=1331):#有默认值的属性必须放在没默认值属性的后面
        self.n_companies = n_companies
        self.n_actions = n_actions
        super().__init__()#调用tf.keras.Model的类初始化方法
        self.dense1 = tf.keras.layers.Dense(units=60, activation=tf.nn.relu)#输入层
        self.dense2 = tf.keras.layers.Dense(units=60, activation=tf.nn.relu)#一个隐藏层
        self.dense3 = tf.keras.layers.Dense(units=60, activation=tf.nn.relu)
        self.dense4 = tf.keras.layers.Dense(units=60, activation=tf.nn.relu)
        self.dense5 = tf.keras.layers.Dense(units=60, activation=tf.nn.relu)
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
    state = sklearn.preprocessing.scale(state)#数据标准化一下
    state = tf.concat((state.flatten(),[capital],[frametime],mean_invested,max),-1)#把降好维的state和capital与frametime连在一起，此时是412长度的一维张量
    state = tf.reshape(state,(1,18))
    return state

 
if __name__ == "__main__":
    start0 = time.time()
    summary_writer = tf.summary.create_file_writer('./tensorboard2') #在代码所在文件夹同目录下创建tensorboard文件夹（本代码在jupyternotbook里跑，所以在jupyternotebook里可以看到）
    #########设置超参数
    learning_rate = 0.0001#学习率
    opt = tf.keras.optimizers.RMSprop(learning_rate)#设定最优化方法
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
    weights_path = 'D:\\data\\eval_Q_weights_mini.ckpt'
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
                    epsilon = epsilon-0.02#也就是经过50万次转移epsilon降到0
                state = jiangwei(state,capital,bianpan_env.mean_invested)#先降维，并整理形状，把capital放进去
                action_index = eval_Q.predict(state)[0]#获得行动q_value
                if random.random() < epsilon:#如果落在随机区域
                    action = random.choice(range(0,1331))#action是一个坐标
                else:
                    action = action_index#否则按着贪心选，这里[0]是因为predict返回的是一个单元素列表
                revenue = bianpan_env.revenue(actions_table[action])#根据行动和是否终赔计算收益
                if frametime == 0:
                    with summary_writer.as_default():
                        tf.summary.scalar('Zinsen',bianpan_env.get_zinsen(),step = bisai_counter)
                        tf.summary.scalar('rest_capital',bianpan_env.gesamt_revenue+500,step = bisai_counter)
                        tf.summary.scalar('wrong_action_rate',bianpan_env.wrong_action_counter/bianpan_env.action_counter,step = bisai_counter)
                        tf.summary.scalar('investion_rate',bianpan_env.gesamt_touzi/500.0,step = bisai_counter)
                next_state,frametime,done,next_capital = bianpan_env.get_state()#获得下一个状态,终止状态的next_state为0矩阵
                #这里需要标识一下终止状态，钱花光了就终止了
                if done:#如果终盘了，跳出
                    replay_buffer.append((state, action, revenue,jiangwei(next_state,next_capital,bianpan_env.mean_invested),1))
                    break
                else:#如果没终盘
                    replay_buffer.append((state, action, revenue,jiangwei(next_state,next_capital,bianpan_env.mean_invested),0))
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
                        loss =  tf.keras.losses.mean_squared_error(y_true = batch_revenue+tf.reduce_max(tf.squeeze(target_Q(np.array(batch_next_state))),axis = -1)*(1-np.array(batch_done))
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

