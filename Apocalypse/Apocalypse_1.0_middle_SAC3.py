#本文件是SAC3模型
#本模型决定取消在网络中过滤不满足条件的行动，而只是将不满足条件的行动的收益赋予0收益
#本模型是SAC3模型，与第2版的区别在于行动变成单位变成0.2————20201020
#此外，取消时间点的限制
#然后更新频率，让actor和critic保持一致，也就是二者共享同一个memory————20201022

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
            revenue = 0.#此处设为0
        if action ==[0.,0.,0.]:
            revenue = 0
            self.no_action_counter+=1#计算无行动率
        #计算本次行动的收益
        return revenue
       
    def get_state(self):
        next_state=self.episode.__next__()
        done = False
        if self.frametime ==0:
            done = True
        return next_state,self.frametime,done,self.capital#网络从此取出下一幕
    
    def get_zinsen(self):
        self.gesamt_touzi =500.0-self.capital
        zinsen  = float(self.gesamt_revenue)/float(self.gesamt_touzi+0.000001)
        return zinsen#这里必须是500.0，否则出来的是结果自动取整数部分，也就是0
    
        
class SumTree(object):

    data_pointer = 0#数据指针，作为存储数据的那个向量self.data的位置指针,初始化为0，即从第一个空位存起

    def __init__(self, capacity):#树的capacity即为回放区的大小，即放在回放区样本的个数
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)#树由一个一维向量组成，每个元素存储一个节点的p，每个外部节点（叶节点）对应一次转移的p，初始化所有的p=0
        #二叉树性质：对于一棵满二叉树，如果外部节点（叶节点）的个数为n，则内部节点的个数为n-1
        #所以节点个数如下：                                                                         
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        #是一个一维向量，用来存储每一次转移，所以每个元素是一次转移，数据类型为对象，所以大小=回放区容量
        #实际上self.data和后self.capacity个叶节点一一对应
        # [--------------data frame-------------]
        #             size: capacity
    
    # 当有新 sample 时, 添加进 tree 和 data
    def add(self, p, data):#给出一次转移和其对应的p，向树上增加节点
        tree_idx = self.data_pointer + self.capacity - 1#tree_idx是该次转移的p的存储的位置，之所以加上self.capacity-1是因为要跳过前面的self.capacity-1个父节点
        self.data[self.data_pointer] = data  # update data_frame，在指针对应的位置放入data
        self.update(tree_idx, p)  # update tree_frame，把data所对应的叶节点的p更新，同时更新与之相关联向上的所有父节点的p
        self.data_pointer += 1#数据指针指向下一个位置
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity，如果指针指到self.data的末尾了，即tree_idx到了self.tree的末尾，即回放区装满了，则重新开始
            self.data_pointer = 0
    # 当 sample 被 train, 有了新的 TD-error, 就在 tree 中更新
    def update(self, tree_idx, p):#给出叶节点的位置和p，更新树
        change = p - self.tree[tree_idx]#得到新p和旧p的差
        self.tree[tree_idx] = p#给新p赋给那个叶节点
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            #由于叶节点的p变化了，于是响应的其所有相关联的父节点的p都要更新，直到更新到根节点（tree_idx = 0）
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):#单个样本抽取（采样）过程
        parent_idx = 0#从根节点开始找起
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1 #该父节点左边的子节点对应在tree向量里的位置        # this leaf's left and right kids，由于是用一维向量存储的数，所以各个点的相对位置由这样的公式确定
            cr_idx = cl_idx + 1#该父节点右边的子节点对应在tree向量里的位置
            if cl_idx >= len(self.tree):        # reach bottom, end search
                #如果计算出的左节点位置超出了树的长度，即parent_idx >= capacity - 1（父节点总共n-1个，则最大坐标为n-2），即父节点已经是叶节点了
                leaf_idx = parent_idx#则此时就已经到达那个叶节点了
                break
            else:       # downward search, always search for a higher priority node,如果没到底，则继续向下搜索
                if v <= self.tree[cl_idx]:#如果v小于等于左子节点的p
                    parent_idx = cl_idx#则在左子节点向下搜索
                else:#如果v大于左子节点的p
                    v -= self.tree[cl_idx]#则v减去左子节点的p更新为新的v
                    parent_idx = cr_idx#并以右子节点为父节点

        data_idx = leaf_idx - self.capacity + 1#该叶节点在数中的位置对应转换成在data里的位置
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]#返回这个样本的叶节点坐标，p和转移数据

    @property#python内置装饰器，把方法作为属性调用，即可以直接调用total_p获得根节点的p值
    def total_p(self):
        return self.tree[0]  # the root


class Critic_Memory(object):  # stored as ( s, a, r, s_ ) in SumTree，一个记忆回放区的类，里面就是一棵书，以及和环境交互的抽样和p值计算方法
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.000025
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):#记忆回放区就是一棵树，存储着记忆数据和其对应的p以及整个树上的p，(p/total_p)即为某个样本被抽中的概率
        self.tree = SumTree(capacity)

    def store(self, transition):#把一次转移交给memory并给它赋予p值然后存储起来
        max_p = np.max(self.tree.tree[-self.tree.capacity:])#在树的所有叶节点（即从后数capacity个及其后面所有的节点）上找到最大的p值
        if max_p == 0:#如果最大叶节点的p值为0，即树是空的
            max_p = self.abs_err_upper#则把最大的p值定为1.0
        self.tree.add(max_p, transition)#以已存在的最大p为p，在树中增加该次转移   # set the max p for new p，
    #抽取每次训练的batch
    def sample(self, n):#每次抽样的过程，其中n为batch_size,即抽出的样本个数
        #b_idx存储的是抽取的batch在tree中的位置index，(n,)是形状参数，即创建一个一维的有n个元素的数组，其中元素由于empty机制是随机数而不是空
        #b_memory存储的是抽取的batch的数据，形状参数为(batch_size,data的长度)，即一个【batch_size*每次转移存储进去的指标数(比如state,capital,next_state....)】形式的矩阵
        #ISWeights是Importance Sampling Weights，即重要度抽样权重，是一个batch_size*1的二维数组，每个元素是一个长度为1的一维数组
        #ISWeights是用于修改损失函数，需要考虑权重，即拥有更大权重的样本的TD-error对总loss贡献更大
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, len(self.tree.data[0])),dtype = object), np.empty((n, 1))
        pri_seg = self.tree.total_p/n#把p按着总p平均分成batch_size份       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1
        p_array = self.tree.tree[-self.tree.capacity:]#每个样本的p值
        nonzero_index = np.nonzero(p_array)#所有非零元素的索引
        #用最小非零p除以总的p，得到min_prob
        min_prob = np.min(p_array[nonzero_index])/self.tree.total_p #所有样本中最小被抽取的概率，即所有的p/total_p中的最小值    # for later calculate ISweight
        for i in range(n):#在每个被用p分出的份儿中随机选取一个v
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)#在每个被用p分出的份儿中随机选取一个v
            idx, p, data = self.tree.get_leaf(v)#根据v获得被抽出来的样本的位置，p和数据值
            prob = p/self.tree.total_p#计算该样本被抽出来的概率
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)#np.power是求幂函数，每个样本的ISweights = (其被抽概率/整个树中最小被抽取概率)^beta
            b_idx[i] = idx
            b_memory[i,:] = data#把这个样本的位置，数据值作为batch里对应的元素
        return b_idx, b_memory, ISWeights
    # train 完被抽取的 samples 后更新在 tree 中的 sample 的 priority
    def batch_update(self, tree_idx, abs_errors):#这里的tree_idx和abs_errors分别是一个有batch_size个元素的列表
        abs_errors += self.epsilon  # convert to abs and avoid 0，纯粹是为了避免td-error（绝对值误差）为0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)#限制td-error，如果td-error大于1，则使其最大为1
        ps = np.power(clipped_errors, self.alpha)#根据训练后的td-error和alpha计算出batch中各样本新的p
        for ti, p in zip(tree_idx, ps):#按着样本在书中的位置
            self.tree.update(ti, p)#更新对应的整个树的p

class Actor_Memory(object):#建立一个演员的当前回合记忆，不过每一新回合开始都清空
    def __init__(self):
        self.memory = deque()#建立储存区
    def store(self,transition):#把每次的转移传给它，这个转移里只包含state,capital和action
        self.memory.append(transition)
    def get_memory(self):
        return self.memory#把记忆还给它
    def clear(self):
        self.memory = deque()

class Q_Network(tf.keras.Model):#给critic定义的q网络
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
        self.dense6_a = tf.keras.layers.Dense(units=self.n_actions)#输出层代表着在当前最大赔率前，买和不买的六种行动的价值

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
        a = self.dense6_a(x)
        q_values = v+(a-tf.reduce_mean(a, axis=1, keepdims=True))#Dueling DQN
        return q_values#q_value是一个（1,4）的张量


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


class Policy_Network(tf.keras.Model):#给actor定义的policy网络
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
        parameters = self.dense6(x)
        return parameters#parameters是一个（1,4）的张量，是决定随机策略分布的参数向量

    def possibility(self,state):#用来对应动作
        parameters = self.call(state)#先根据jiangwei好的state求参数向量
        possibilities = tf.nn.softmax(parameters)#把这些动作改成概率
        return possibilities#返回可选行动及其概率
    

class Actor(object):
    def __init__(self,lr=0.0003):
        self.net = Policy_Network()#初始化网络
        self.opt = tf.keras.optimizers.Adam(lr,amsgrad=True)#设定最优化方法
        self.opt_alpha = tf.keras.optimizers.Adam(lr,amsgrad=True)#参考论文
        self.memory = Actor_Memory()
        self.alpha = tf.Variable(initial_value=0.2)#参考tf2rl 
        self.target_alpha = -np.log((1.0 / 4)) * 0.98

    def choose_action(self,state):
        possibilities = self.net.possibility(state)#获得行动概率二元组并解耦
        index = np.random.choice(range(4), p=np.array(possibilities).ravel())#根据概率选择索引
        return index

    def learn(self):#把当前回合的记忆和critic算出的td_error传给它
        memory = self.memory.get_memory()
        batch_state, batch_capital,batch_next_capital,batch_action, batch_revenue, batch_next_state ,batch_done = zip(*memory)#把本回合的转移拆成两个batch
        with tf.GradientTape(persistent=True) as tape: 
            q = np.array(list(map(critic.target_Q,batch_state)))#所有4个动作的Q值
            prob = tf.nn.softmax(self.net(tf.squeeze(batch_state)))#所有动作的概率
            log_prob = tf.math.log(prob)
            loss = tf.reduce_mean(tf.reduce_sum(prob*(self.alpha*log_prob-tf.squeeze(q)),axis = 1))
            loss_alpha = tf.reduce_mean(tf.reduce_sum(prob*(-self.alpha*log_prob+self.target_alpha),axis = 1))
        grads = tape.gradient(loss, self.net.variables)
        grads_alpha = tape.gradient(loss_alpha, [self.alpha])
        self.opt.apply_gradients(grads_and_vars=zip(grads, self.net.variables))#更新策略
        self.opt_alpha.apply_gradients(grads_and_vars=zip(grads_alpha, [self.alpha]))#更新alpha
        del tape
        self.net.save_weights(actor_weights_path, overwrite=True)
        return loss

        
        
class Critic(object):#只需要做每次学习，以及把相应的td_error传给Actor
    def __init__(self,lr=0.0003):
        self.local_Q = Q_Network()#按论文中写的local_Q
        self.target_Q = Q_Network()#按论文中写的target_Q
        self.gamma = 0.99999
        self.memory_size = 1000000#按论文中给的1e6大小
        self.batch_size=500
        self.memory = Critic_Memory(capacity=self.memory_size)
        self.opt1 = tf.keras.optimizers.Adam(lr,amsgrad=True)#设定最优化方法
        self.opt2= tf.keras.optimizers.Adam(lr,amsgrad=True)
        self.target_repalce_counter = 0
        self.tau = 5e-3#tf2rl用的这个数
    
    def learn(self):
        tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)
        batch_state, batch_capital,batch_next_capital,batch_action, batch_revenue, batch_next_state ,batch_done = zip(*batch_memory)
        with tf.GradientTape(persistent=True) as tape:
            #因为用了两个网络，所以y_true要分别求，先求local_Q的
            all_q1 = tf.squeeze(list(map(critic.local_Q,batch_state)))#获得此刻状态的所有4个动作的q值
            one_hot_matrix = tf.one_hot(np.array(batch_action),depth=4,on_value=1.0,off_value=0.0)#有batch_size行，4列
            y_true1 = tf.reduce_sum(all_q1*one_hot_matrix,axis=1)#获得此刻状态的对应动作的q值
            next_all_q1 = tf.squeeze(list(map(critic.local_Q,batch_next_state)))#获得下一时刻所有动作的q值
            next_prob1 =tf.nn.softmax(actor.net(tf.squeeze(batch_next_state)))#得到下一个满足条件动作的概率分布
            next_v1 = next_all_q1-actor.alpha*tf.math.log(next_prob1)#得到下一状态各种动作下的v
            expectation_v1 = tf.reduce_sum(next_prob1*next_v1,axis=1)#获得下一个状态的v的期望.
            y_pred1 = batch_revenue+self.gamma*expectation_v1*(1-np.array(batch_done))
            loss1 = tf.reduce_mean(ISWeights * tf.math.squared_difference(y_true1, y_pred1))
            #下面算target_Q的
            all_q2 = tf.squeeze(list(map(self.target_Q,batch_state)))#获得此刻状态的所有4个动作的q值
            one_hot_matrix = tf.one_hot(np.array(batch_action),depth=4,on_value=1.0,off_value=0.0)#有batch_size行，4列
            q2 = tf.reduce_sum(all_q2*one_hot_matrix,axis=1)#获得此刻状态的对应动作的q值
            next_all_q2 = tf.squeeze(list(map(self.target_Q,batch_next_state)))#获得下一时刻所有动作的q值
            next_prob2 =tf.nn.softmax(actor.net(tf.squeeze(batch_next_state)))#得到下一个满足条件动作的概率分布
            log_next_prob2 = tf.math.log(next_prob2)
            next_v2 = next_all_q2-actor.alpha*log_next_prob2#得到下一状态各种动作下的v
            expectation_v2 = tf.reduce_sum(next_prob2*next_v2,axis=1)#获得下一个状态的v的期望.
            y_pred2 = batch_revenue+self.gamma*expectation_v2*(1-np.array(batch_done))#终止状态的v为0
            y_true2 = q2
            loss2 = tf.reduce_mean(ISWeights * tf.math.squared_difference(y_true2, y_pred2))
            abs_errors = tf.abs(y_true2 - y_pred2)
        grads1 = tape.gradient(loss1, self.local_Q.variables)
        grads2 = tape.gradient(loss2, self.target_Q.variables)
        self.memory.batch_update(tree_idx, abs_errors)#用target_Q的td_error更新tree
        self.opt1.apply_gradients(grads_and_vars=zip(grads1, self.local_Q.variables))#更新参数
        self.opt2.apply_gradients(grads_and_vars=zip(grads2, self.target_Q.variables))#更新参数
        del tape
        self.target_Q.set_weights(self.local_Q.get_weights()*self.tau+self.target_Q.get_weights()*(1-self.tau))
        return loss2#返回loss好可以记录下来输出








if __name__ == "__main__":
    summary_writer = tf.summary.create_file_writer('./tensorboard_1.0_middle_SAC3')
    summary_writer2 = tf.summary.create_file_writer('./tensorboard_1.0_middle_SAC3/use_out_time')
    summary_writer3 = tf.summary.create_file_writer('./tensorboard_1.0_middle_SAC3/max_frametime')
    summary_writer4 = tf.summary.create_file_writer('./tensorboard_1.0_middle_SAC3/used_steps')
    summary_writer5 = tf.summary.create_file_writer('./tensorboard_1.0_middle_SAC3/bisai_steps')
    summary_writer6 = tf.summary.create_file_writer('./tensorboard_1.0_middle_SAC3/actor_loss')
    summary_writer7 = tf.summary.create_file_writer('./tensorboard_1.0_middle_SAC3/critic_loss')
    summary_writer8 = tf.summary.create_file_writer('./tensorboard_1.0_middle_SAC3/mini_critic_loss')
    start0 = time.time()
    epsilon = 1.            # 探索起始时的探索率
    #final_epsilon = 0.01            # 探索终止时的探索率
    resultlist = pd.read_csv('D:\\data\\results_20141130-20160630.csv',index_col = 0)#得到赛果和比赛ID的对应表
    actions_table = [[0.,0.,0.],[0.2,0.,0.],[0.,0.2,0.],[0.,0.,0.2]]#给神经网络输出层对应一个行动表
    step_counter = 0
    learn_step_counter = 0
    target_repalce_counter = 0 
    bisai_counter = 1
    critic_weights_path = 'D:\\data\\critic_Q_weights_1.0_middle_SAC3.ckpt'
    actor_weights_path = 'D:\\data\\actor_weights_1.0_middle_SAC3.ckpt'
    filefolderlist = os.listdir('F:\\cleaned_data_20141130-20160630')
    actor = Actor()#实例化一个actor
    #actor.net.load_weights(pre_weights_path)#读入1.0_sofort2的权重
    critic = Critic()#实例化一个critic
    #critic.eval_Q.load_weights(pre_weights_path)#读入1.0_sofort2的权重
    #critic.target_Q.load_weights(pre_weights_path)#读入1.0_sofort2的权重
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
            #actor.memory.clear()#每场比赛开始前要清空记忆————在SAC3中取消了
            state,frametime,done,capital =  bianpan_env.get_state()#把第一个状态作为初始化状态
            max_frametime = bianpan_env.max_frametime#得到本场比赛最大的frametime
            end_switch = False
            bisai_steps = 0
            used_steps = 0
            while True:
                step_counter+=1#每转移一次，步数+1
                if step_counter>=2000000:
                    epsilon = 0.#如果超过200万次转移，转贪心
                state = jiangwei(state,capital,frametime,bianpan_env.mean_invested)#先降维，并整理形状，把capital放进去
                if epsilon == 1.:#如果落在随机区域
                    action = random.randint(0,3)
                else:
                    action = actor.choose_action(state)
                revenue = bianpan_env.revenue(actions_table[action])#计算收益
                next_state,next_frametime,done,next_capital = bianpan_env.get_state()#获得下一个状态,终止状态的next_state为0矩阵
                bisai_steps+=1
                if end_switch == False:#如果没花光
                    use_out_time = 1#没花光就当做1
                    used_steps+=1
                if (next_capital<= 0.) and (end_switch == False):
                    use_out_time = frametime
                    end_switch = True
                if(step_counter<=2000):
                    print('已转移'+str(step_counter)+'步')               
                if done:#终盘时储存信息，同时更新actor，清除actor内存
                    transition = np.array((state,capital,next_capital,action, revenue,jiangwei(next_state,next_capital,next_frametime,bianpan_env.mean_invested),1))
                    #actor.memory.store(transition)#actor和critic共享同一记忆
                    critic.memory.store(transition)
                    with summary_writer.as_default():
                        tf.summary.scalar('Zinsen',bianpan_env.get_zinsen(),step = bisai_counter)
                        tf.summary.scalar('rest_capital',bianpan_env.gesamt_revenue+500.,step = bisai_counter)
                        tf.summary.scalar('wrong_action_rate',bianpan_env.wrong_action_counter/bianpan_env.action_counter,step = bisai_counter)
                        tf.summary.scalar('investion_rate',bianpan_env.gesamt_touzi/500.0,step = bisai_counter)
                    with summary_writer2.as_default():
                        tf.summary.scalar('times',use_out_time,step =bisai_counter)
                    with summary_writer3.as_default():
                        tf.summary.scalar('times',bianpan_env.max_frametime,step =bisai_counter)
                    with summary_writer4.as_default():
                        tf.summary.scalar('steps',used_steps,step =bisai_counter)
                    with summary_writer5.as_default():
                        tf.summary.scalar('steps',bisai_steps,step =bisai_counter)
                    with summary_writer2.as_default():
                        tf.summary.scalar('times',use_out_time,step =bisai_counter)
                    with summary_writer3.as_default():
                        tf.summary.scalar('times',bianpan_env.max_frametime,step =bisai_counter)
                    with summary_writer4.as_default():
                        tf.summary.scalar('steps',used_steps,step =bisai_counter)
                    with summary_writer5.as_default():
                        tf.summary.scalar('steps',bisai_steps,step =bisai_counter)
                    state = next_state
                    capital = next_capital
                    frametime = next_frametime
                    break
                else:
                    transition = np.array((state,capital,next_capital,action, revenue,jiangwei(next_state,next_capital,next_frametime,bianpan_env.mean_invested),0))
                    actor.memory.store(transition)
                    critic.memory.store(transition)
                    state = next_state
                    capital = next_capital
                    frametime = next_frametime
                if (step_counter >2000) and (step_counter%50 == 0) :
                    critic_loss = critic.learn()#先critic学习
                    actor_loss = actor.learn()#再actor学习
                    with summary_writer6.as_default():
                        tf.summary.scalar('losses',actor_loss,step = learn_step_counter)
                    with summary_writer7.as_default():
                        tf.summary.scalar('losses',critic_loss,step = learn_step_counter)
                    if learn_step_counter%2000 ==0:
                        critic.target_Q.set_weights(critic.local_Q.get_weights())
                    learn_step_counter+=1#每学习一次，学习步数+1
                    print('critic已学习'+str(learn_step_counter)+'次')
            end=time.time()
            bisai_counter+=1
            print('比赛'+filepath+'已完成'+'\n'+'用时'+str(end-start)+'秒\n')
    end0 = time.time()
    print('20141130-20160630总共用了'+str(end0-start0)+'秒')

