#本文件是记录思路的文件
#由于数据的状态空间是有限的，离散的，所以应该先进行异常值清理，或者把异常值单独分拆出来，用一套不同的策略。
#动作空间要考虑是设成有限离散的还是连续的
#这个问题一个特点就是，行动本身并不会影响环境。
"""
构造状态空间的几种可能的方式：
以0.1为单位，构造一个离散的，有限的，状态空间。
1.直接使用300多个公司在同一场比赛，同一时刻的状态这样一个300*8的矩阵作为该时刻的状态
2.选择主要的5个公司的赔率作为该时刻的状态
3.用降维方法把数据转化成5*8的矩阵
4.又或者把通过降维减少赔率指标的个数，或者构造新的综合的指标

动作空间：
{。。。买入1公司上盘+50，买入1公司上盘+45。。。买入1公司上盘+5，不作为0，买入1公司下盘+5，买入1公司下盘+10。。。。。。}，单位欧元
                                                +
{。。。买入2公司上盘+50，买入2公司上盘+45。。。买入2公司上盘+5，不作为0，买入2公司下盘+5，买入2公司下盘+10。。。。。。}
                                               +
                                               。
                                               。
                                               。
"""
'''
Apocalypse 1.0 策略思想：
300个公司的赔率其实可以组合成一个最优赔率，比如用A公司的胜，B公司的平和C公司的负，可以构成一个水位，
这样就可以把300个公司的这样一个矩阵转化成两个向量，一个最低水位，一个最高水位，这两个向量以及距离开赛时间共同组成了状态向量。
如果加入其它组合形式在状态里，共有300*300*300种组合，那么图像化表示就像一个各处密度不同的棒棒，棒棒内的密度越来越往高水位涨，那么最终会推动状态向量发生改变。
每当同一个比赛有公司变盘，那这300个公司提取出的水位向量就有可能发生变化。
首先，在单个时刻最高水位和最低水位之间，就有可能有套利的空间。
第二，可以利用机器学习，利用预测之后状态的变化。
第三，在状态发生转移之前（即更高的水位或更低的水位出现之前），其实300个公司的赔率内部可能已经发生变化，只是没有引起状态向量的变化，这一点也可以用作预测。
第四，为了防止只能学习到跨平台套利的策略，应该可以采用某种试探型策略，来鼓励在当下花钱，但是在为了可能盈利的策略。
第五，给每个回合（每场比赛从开盘到比赛开始）一个固定的可用资金量，策略则是用百分比表示投入的金额，比如5000欧，然后每次5欧，即1%为一个单位
第六，假设不同的比赛只要赔率开盘情况相同，那么之后的马尔科夫转移过程也相同。
第七，暂时不考虑具体球队过往战绩，主客场，过往赔率的影响。
第八，暂时不考虑同一段时间市场的状态（比如有多场比赛是同时进行变盘的，那么这段时间里，可能整个市场就倾向往高水位走）
'''
'''
Apocalypse 2.0 策略思想：
1.预测中加入过往战绩，或者赔率变化，球队，主客场，赛季，季节的影响。
2.尝试先学习分布，再进行动态规划。
3.或者先用神经网络预训练出一个模型，再进行Dyna-Q
'''

'''
Apocalypse 1.0 数据清洗过程：
1.先选出一年的数据集
2.把赔率状态换算成密度棒的形式，当然密度棒内的单个组合要连接上具体哪个公司，以方便策略进行选择买入
3.把每一场比赛都做成分幕式的赔率变化，长度不一。
4.密度棒要带上距离比赛前的时间，以确定终止状态。
'''

'''
Apocalypse 1.0系列 可能模型：
0.0 DQN，Dueling DQN（竞争构架Q网络），DRQN（深度循环Q网络，把DQN的全连接层换成LSTM），DDPG（连续动作空间的算法）
1.0 Dyna-Q(强化学习第8章)
1.1 TD(λ)，然后价值函数用神经网络逼近，即TD-Gammon（强化学习第12章）
1.2 策略梯度，策略用神经网络，价值函数也用神经网络（强化学习第13章）
'''



