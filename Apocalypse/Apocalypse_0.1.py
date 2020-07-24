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
class Dataloader():#需要定义一个数据预处理器，除了对输入数据处理外，还要把策略的空间搞出来，在所有的公司随机选
    def __init__(self):
        #传入原始数据，为一个不定长张量对象
        print('数据预处理器初始化完成')
        

    def load(self,origin):
        self.origin=origin




        return #返回固定尺寸张量






class Q_Model(tf.keras.Model):
    def __init__(self,
                      n_companies,
                      n_features,
                      n_actions):
        self.n_companies = n_companies
        self.n_features = n_features
        self.n_actions = n_actions
        super().__init__()#调用tf.keras.Model的类初始化方法
        self.flatten = tf.keras.layers.Flatten() #把单个矩阵展平
        self.dense1 = tf.keras.layers.Dense(units=int(3*self.n_companies*self.n_features), activation=tf.nn.relu)#输入层
        self.dense2 = tf.keras.layers.Dense(units=int(0.75*self.n_companies*self.n_features), activation=tf.nn.relu)#一个隐藏层
        self.dense3 = tf.keras.layers.Dense(units=6)#输出层代表着在当前最大赔率前，买和不买的六种行动的价值

    def call(self,inputs): #输入
        x = self.flatten(inputs)#输出[2100,1]
        x = self.dense1(x)#输出[6300,1]
        x = self.dense2(x)#输出= [4725,1]
        output = self.dense3(x)#输出= [6,1]
        return output


         
         








class Decision_maker():
    def __init__(self,q_value_vector):
        self.q_value_vector = q_value_vector

    def maker(self):#用来根据q_value找到响应的公司，然后根据已买入的情况出一个决策


        return    
        