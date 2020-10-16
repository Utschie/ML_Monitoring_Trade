
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
'''
mini+transpose+TruncatedSVD+标准化
输出数 = 18
'''
def jiangwei_mini(state,capital,mean_invested):
    tsvd = TruncatedSVD(1)
    max_host = state[tf.argmax(state)[2].numpy()][2]
    max_fair = state[tf.argmax(state)[3].numpy()][3]
    max_guest = state[tf.argmax(state)[4].numpy()][4]
    max = [max_host,max_fair,max_guest]
    frametime = state[0][0]#取出frametime时间
    state=np.delete(state, 0, axis=-1)#把frametime去掉，则state变成了（410,7）的矩阵
    state = tsvd.fit_transform(np.transpose(state))#降维成（7,1）的矩阵
    state = sklearn.preprocessing.scale(state)#数据标准化一下
    state = tf.concat((state.flatten(),[capital],[frametime],mean_invested,max),-1)#7+1+1+6+3=18
    state = tf.reshape(state,(1,18))
    return state


def jiangwei_mini(state,capital,frametime,mean_invested):
    invested = [0.,0.,0.,0.,0.,0.]
    max_host = state[tf.argmax(state)[2].numpy()][2]
    max_fair = state[tf.argmax(state)[3].numpy()][3]
    max_guest = state[tf.argmax(state)[4].numpy()][4]
    max = [max_host,max_fair,max_guest]
    tsvd = TruncatedSVD(1)
    frametime = frametime/50000.0
    length = len(state)/410.0#出赔率的公司数归一化
    invested[0] = mean_invested[0]/25.0
    invested[1] = mean_invested[1]/500.0
    invested[2] = mean_invested[2]/25.0
    invested[3] = mean_invested[3]/500.0
    invested[4] = mean_invested[4]/25.0
    invested[5] = mean_invested[5]/500.0
    state=np.delete(state, 0, axis=-1)
    state = tsvd.fit_transform(np.transpose(state))#降维成（1,7）的矩阵
    state = tf.concat((state.flatten(),[capital/500.0],[frametime],invested,[length],max),-1)#7+1+1+6+1+3=19
    state = tf.reshape(state,(1,19))
    return state



'''
TPU+TruncatedSVD+标准化
输出数 = 421
'''
def jiangwei_TPU(state,capital,mean_invested):
    tsvd = TruncatedSVD(1)
    max_host = state[tf.argmax(state)[2].numpy()][2]
    max_fair = state[tf.argmax(state)[3].numpy()][3]
    max_guest = state[tf.argmax(state)[4].numpy()][4]
    max = [max_host,max_fair,max_guest]
    frametime = state[0][0]#取出frametime时间
    state=np.delete(state, 0, axis=-1)#把frametime去掉，则state变成了（410,7）的矩阵
    state = tsvd.fit_transform(state)#降维成（410,1）的矩阵
    state = sklearn.preprocessing.scale(state)#数据标准化一下
    state = tf.concat((state.flatten(),[capital],[frametime],mean_invested,max),-1)#把降好维的state和capital与frametime连在一起，此时是412长度的一维张量
    state = tf.reshape(state,(1,421))
    return state




'''
分位数,需配合变大小statematrix的Env使用,最终大小可调
'''
def jiangwei_middle(state,capital,mean_invested):
    frametime = state[0][0]
    state=np.delete(state, 0, axis=-1)
    length = len(state)
    percentile = np.vstack(np.percentile(state,i,axis = 0)[1:4] for i in range(0,105,5))#把当前状态的0%-100%分位数放到一个矩阵里
    state = tf.concat((percentile.flatten(),[capital],[frametime],mean_invested,[length]),-1)
    state = tf.reshape(state,(1,72))#63个分位数数据+8个capital,frametime和mean_invested,共72个输入
    return state
    
