#先用简单的Q学习，即先不对赔率变化建立模型
#需要一个序列生成器用来把某一场比赛的分幕状态分步传入策略选择器
#需要一个策略选择器根据传入的一步的状态以及当前的动作价值函数Q和策略用来生成行动数据
#需要一个策略更新器用来在每一步下更新策略选择器中的动作价值函数，这里涉及到利用当前动作价值函数选择价值最大的动作的一个问题
#Q网络把当前状态（即每个时刻对应的那张表）和当前动作作为x，把由下一状态和当前回报组成的tagaet值作为标签y，然后使损失函数最小
#算法见https://zhuanlan.zhihu.com/p/21421729
#莫烦的github上用的是Nature Q网络，即单独做了一个Target Q网络
#神经网络传入的数据是一个3维张量，batch_size*公司数*指标数，但是由于更新时间差异，公司数会有所不同，就好像图像尺寸大小不一一样
#上述情况或者是把其他未更新数据用东西填充，或者是把公司数*指标数这样一个当前状态的矩阵转化成一个固定大小的东西，比如“密度棒”
#0.1版本先用0元素填充方式，不管用什么填充，反正没信息的那个节点不能被激活
#把公司id按从小到大排列然后组成一个公司数*指标数的矩阵，保证传入的每个矩阵的对应行都对应相同的公司
#另外给定的环境模型除了赔率矩阵外，还要有终止状态，一个就是钱花完了的状态，还有一个就是终赔出来的状态
#还是应该先数据预处理，处理成一个水位的n_company*1的向量，然后根据得出的q值解码出怎么买，来减小策略空间的数量————20200724
#每一步转移的状态可能与之前的选择有关，因为如果在某几买入了胜和负，那么。。。。需要研究一下————20200724
#还是应该做一个环境类，模型跟其互动时，要返回状态转移还要返回收益————20200725
#只要(1/win+1/fair+1/lost)<1,这就有利可图，假设投资比例分别，所以要让1/win+1/fair+1/lost尽量小，也就是希望win，fair和lost尽量大————20200724
#第一，要让个买入的赔率尽量大；第二，要让投资配比的总收益率尽量高或者尽量稳————20200724
#第三，要让一次比赛的操作步骤尽量少；第四，要用整数
#所以q函数应该决定，要不要买入这里面的赔率最大值，然后还需要一个策略，就是买多少的策略。
#最后q函数的值应该是还是代表着行动价值，q函数出选择，然后还需要定义一个决策器，用来接收q函数的决策计算要买多少————20200724
#其实可以把后面的决策器部分也合并进来，这样就把最后一层节点就会很多，所以本版先把决策器分出来，确定好买入卖出决策后，具体数量就用线性方程组求，使收益恒定
#记忆回溯部分之后或许可以尝试利用统计距离选取与当前状态最相近的50个训练
#批量梯度下降是在损失函数迭代的过程中算法不同，神经网络结构本身不需要把batch_size当做一个维度————20200724
#需要知道所有数据中有多少个cid，这就很烦，因为你也不知道是不是有新公司之类的，而且一年的数据就已经很多了，遍历一遍很讨厌————20200726
#到16年6月30为止总共有410个cid
#妈的当初爬数据要是把赛果也爬下来就好了
#是否为终止状态应该由statematrix里的frametime是否为0决定————20200727
#Q函数设置有误，首先Q网络输出策略不是6种，而是2^3=8种，或者最高赔率下每个都有投0,5,10,……,50，11种动作，则策略集为11^3=1331种策略————20200727
#然后应该给策略做一个表格，然后按照index进行收益的计算————20200727
#损失函数设置错误，在损失函数中，r是当前的随机贪心策略出，而q_target则是由最优策略出，q_eval是当前动作的那个q值而不是全部动作的q值————20200727
#当前的网络参数是爆炸的，参数数量高达5900多万————20200729
#20141130-20160630这段时间共有7500多万次转移，考虑到样本至少为参数数量的10倍，所以参数数量要控制在750万以下————20200727
#所以还是要数据降维————20200727
#过段时间要把赛果爬出来，至少是20141130-20160630的，把比赛代号和赛果配对起来（已完成）
#状态虽然是个列表，但是传入前应该再加一位，即变成（1,410,8）的张量，才能正常展平张开，切记！！！——————20200729
#invested——已投入资本可以有两种算法，一是用一个求平均的向量存储，二是把每一次的交易存储，然后最后进行计算，感觉后一次比较好————20200729
#计算器写是大概写完了，但还是感觉程序执行顺序有点儿不对，状态和收益的对应关系好像错开了————20200730（好像又没啥问题）
#但是要保证终止状态的收益永远为0，可能代表终止状态的矩阵需要换个矩阵来表示————20200730（已解决，把第一个状态作为初始化状态就可以了）
#参数更新过程要注意让终止状态0矩阵的收益为0————20200730(已解决)
#不对，中间钱花光了跳出去的也应该结算一次(已解决)
#或许不需要中间结算，只要capital小于0了，那之后的一律无法行动，这样的话，capital也可以作为一个状态变量传入神经网络————20200731
#最后出可视化的时候应该把算一下收益率————20200731
#环境的结算方法可能存在一些问题————20200731
#只在终盘时结算，其他时候收益都是负值，因为如果提前跳出会导致浪费样本————20200731
#算revenue时给超出资金量的策略的收益赋予很大的负值，然后把capital也放在变量里送给神经网络(已解决)
#可以直接应用pca降维，在statematrix里去掉frametime和cid之后降到一维，然后再把frametime作为一个单独的值和它连起来放入神经网络
#但是env里也必须保存原statematrix，就在神经网络里增加降维这一项就好了
#这样就不需要flatten层了，然后也不需要把statematrix做成3维张量了
#经过降维，把参数降到了1,295,855个————20200731
#现在暂时采用这样6层，每层单元数740多个，参数数量为350万————20200731
#全连接层的FLOPs算法就是各层的2*输入数*输出数 = 6991124的FLOPs，即一个样本通过神经网络需要的计算量————20200731
#每次迭代需要batch_size,则50*700M = 35000MFLOPs，假设进行7500万次迭代，则训练整个模型至少需要35000M*75M = 2.625*10^18次FLOPs————20200802
#这还不考虑每次迭代的最优化过程。960M显卡的算力是5.0TFLOPS，即5*10^12次FLOP每秒，则跑完至少需要525000秒，即8750分钟即146小时即6天多————20200802
#尝试写成用TPU跑的版本————之后要写成TPU跑的版本
#可以在jupyter notebook 进行测试了，首先要%load_ext tensorboard和%tensorboard --logdir=./tensorboard 从而让jupyter notebook可以使用tensorboard
#好像如果神经网络重写后设定了两个参数，那所谓的batch_size这一项就不存在了，所以应该把capital和state连起来
#降维活动应该在神经网络外进行（已解决）
#在修修补补之后终于可以破破烂烂地跑起来了————20200803
#刚刚把minibatch的size改成500，不知道怎么样————20200803 11:39
#minibatch调成500用cpu跑会变很慢
#由于tensorboard显示不全，所以改成一场比赛画一个点
#self.gesamt_revenue以及收益率的计算有些问题，因为把所有的都计入了，连钱不够的也计入了————20200803(已解决)
#rest_capital应该在self.gesamt_revenue的基础上加500，来表示最后剩下了多少钱，比如gesamt_revenue=-500,则剩下了0————20200804
#先改成GPU算法小测一下这一天的速度，但是GPU用不了（已解决）
#那就试试TPU如何————20200806
#前面1000次转移差不多用了45-50秒，也就是说如果没有学习的部分，差不多80分钟就完成一天了————20200806
#现阶段重点是提升性能，利用并行，tf.data,Intel的mkl等等工具进行大概，现在的cpu使用率只有最多30%，gpu只有12%，太慢————20200806
'''
两种可能：第一种是把矩阵数据预处理成一个向量，然后输出一个向量再解码成策略
         第二种是前面输入数据不用处理成向量，然后后面的q值函数处理成一个向量，然后把这个向量解码成策略
最重要的其实是找到一个可能的q值函数和策略的对应关系，因为每个状态其实都有一个最高水位和最低水位，所以其实每次的动作只有6*11种可能
即在最高/最低水位买入胜/平/负0,10,20...100元
'''
'''
批量梯度下降方法如下
opt = tf.keras.optimizers.RMSprop()  设定最优化方法
var_list=......
loss=。。。。。。   设定损失函数
其中loss对象要继承tensorflow的基础loss类然后重写，loss类本身就是支持y_true和y_pred是以batch的list传入
opt.minimize( loss, var_list, grad_loss=None, name=None) 传入损失函数，变量列表，然后更新var_list
'''
#先写一个神经网络类
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
            revenue = -500#则收益是个很大的负值（正常来讲revenue最大-50）
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
   

class Q_Network(tf.keras.Model):
    def __init__(self,
                      n_companies=412,
                      n_actions=1331):#有默认值的属性必须放在没默认值属性的后面
        self.n_companies = n_companies
        self.n_actions = n_actions
        super().__init__()#调用tf.keras.Model的类初始化方法
        self.dense1 = tf.keras.layers.Dense(units=int(1.8*self.n_companies), activation=tf.nn.relu)#输入层
        self.dense2 = tf.keras.layers.Dense(units=int(1.8*self.n_companies), activation=tf.nn.relu)#一个隐藏层
        self.dense3 = tf.keras.layers.Dense(units=int(1.8*self.n_companies), activation=tf.nn.relu)
        self.dense4 = tf.keras.layers.Dense(units=int(1.8*self.n_companies), activation=tf.nn.relu)
        self.dense5 = tf.keras.layers.Dense(units=int(1.8*self.n_companies), activation=tf.nn.relu)
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
    


def jiangwei(state,capital):
    pca = PCA(n_components=1)
    frametime = state[0][0]#取出frametime时间
    state=np.delete(state, 0, axis=-1)#把frametime去掉，则state变成了（410,7）的矩阵
    state = pca.fit_transform(state)#降维成（410,1）的矩阵
    state = tf.concat((state.flatten(),[capital],[frametime]),-1)#把降好维的state和capital与frametime连在一起，此时是412长度的一维张量
    state = tf.reshape(state,(1,412))
    return state

 
         


if __name__ == "__main__":
    start0 = time.time()
    summary_writer = tf.summary.create_file_writer('./tensorboard') #在代码所在文件夹同目录下创建tensorboard文件夹（本代码在jupyternotbook里跑，所以在jupyternotebook里可以看到）
    #########设置超参数
    learning_rate = 0.01#学习率
    epsilon = 1.            # 探索起始时的探索率
    #final_epsilon = 0.01            # 探索终止时的探索率
    batch_size = 50
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
    weights_path = 'D:\\data\\eval_Q_weights.ckpt'
    filelist = os.listdir('D:\\data\\2014-11-30')#读取这一天的文件名
    ################下面是单场比赛的流程



    for i in filelist:#挨场比赛训练
        start=time.time()
        filepath = 'D:\\data\\2014-11-30\\'+i#文件路径
        bisai_id = int(re.findall(r'\\(\d*?).csv',filepath)[0])#从filepath中得到bisai代码的整型数
        result = resultlist.loc[bisai_id]#其中result.host即为主队进球，result.guest则为客队进球
        bianpan_env = Env(filepath,result)#每场比赛做一个环境
        state,frametime,done,capital =  bianpan_env.get_state()#把第一个状态作为初始化状态
        opt = tf.keras.optimizers.RMSprop(learning_rate)#设定最优化方法
        with summary_writer.as_default():
            tf.summary.scalar("Capital", capital,step = bisai_counter)
        while True:
            if step_counter % 1000 ==0:
                epsilon = epsilon-0.01 
                print(epsilon)
            state = jiangwei(state,capital)#先降维，并整理形状，把capital放进去
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
            next_state,frametime,done,next_capital = bianpan_env.get_state()#获得下一个状态,终止状态的next_state为0矩阵
            #这里需要标识一下终止状态，钱花光了就终止了
            if done:#如果终盘了，跳出
                replay_buffer.append((state, action, revenue,jiangwei(next_state,next_capital),1))
                break
            else:#如果没终盘
                replay_buffer.append((state, action, revenue,jiangwei(next_state,next_capital),0))
            state = next_state
            capital = next_capital
            
            #下面是参数更新过程
            if (step_counter >1000) and (step_counter% 10 == 0) :#1000步之后每转移10次进行一次eval_Q的学习
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
                opt.apply_gradients(grads_and_vars=zip(grads, eval_Q.variables))
                learn_step_counter+=1#每学习一次，学习步数+1
                print('已学习'+str(learn_step_counter)+'次')
                if (learn_step_counter % 300 == 0) and (learn_step_counter > 0):#每学习300次，target_Q网络参数进行一次变量替换
                    eval_Q.save_weights(weights_path, overwrite=True)#保存并覆盖之前的检查点，储存权重
                    target_Q.load_weights(weights_path)#读取eval_Q刚刚保存的权重
                    target_repalce_counter+=1
                    print('目标Q网络已更新'+str(target_repalce_counter)+'次')
            step_counter+=1#每转移一次，步数+1
            print(str(step_counter))
        end=time.time()
        bisai_counter+=1
        print('比赛'+filepath+'已完成'+'\n'+'用时'+str(end-start)+'秒\n')
    end0 = time.time()
    print('共用时'+str(end0-start0)+'秒')







    
    
            

        



