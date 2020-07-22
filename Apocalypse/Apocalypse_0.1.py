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


#先写一个神经网络类
import tensorflow as tf
class Dataloader():#需要定义一个数据预处理器，把每次不同大小的状态矩阵转成固定大小
    def __init__(self):
        #传入原始数据，为一个不定长张量对象
        print('数据预处理器初始化完成')
        

    def load(self,origin:tf.RaggedTensor):
        self.origin=origin




        return #返回固定尺寸张量






class Q_Model(tf.keras.Model):
    def __init__(self,batch_size,
                      n_companies,
                      n_features,
                      n_actions):
        self.batch_size = 50
        self.n_companies = n_companies
        self.n_features = n_features
        self.n_actions = n_actions
        super().__init__()#调用tf.keras.Model的类初始化方法
        