#这是datacleaning的完整运行文件,先仅限于2014-11-30.txt这一个文件，测试一下速度
#bisai2excel即便是单场比赛也有5240次变盘，写入excel非常缓慢
#即便转成json，单场比赛写入后的文件居然有400M，因为拆分后每张表的keys都要重复一遍，这样就变得很大
#应该想办法把数据缩小，比如看能不能用多维数组之类的
from gevent import monkey;monkey.patch_all()
import gevent
import re
import datetime
import pandas as pd
import time
import csv
'''先转成csv'''
#进入文件夹，生成文件名序列
f=open('D:\\data\\20141130-20160630\\2014-11-30.txt','r')
line=f.readline()
f.close()#要关闭数据集
#把这一行数据用re转成列表
datalist=re.findall('{.*?}',line)
keys=['date', 'urlnum', 'league', 'cid', 'zhudui', 'kedui', 'companyname', 'resttime', 'fanhuanlv', 
'peilv_host', 'peilv_fair', 'peilv_guest', 
'gailv_host', 'gailv_fair', 'gailv_guest', 
'kailizhishu_host', 'kailizhishu_fair', 'kailizhishu_guest']
df=pd.DataFrame(columns=keys)#先列好字段
#把列表中各个元素转成字典,并且把peilv，gailv和kailizhishu分拆成三列，否则无法正确读入pandas
def str2dict(str):#讲datalist中的单个元素转换插入dataframe的函数
    dic=eval(str)
    del dic['timestamp']
    dic['peilv_host']=dic['peilv'][0]
    dic['peilv_fair']=dic['peilv'][1]
    dic['peilv_guest']=dic['peilv'][2]
    del dic['peilv']
    dic['gailv_host']=dic['gailv'][0]
    dic['gailv_fair']=dic['gailv'][1]
    dic['gailv_guest']=dic['gailv'][2]
    del dic['gailv']
    dic['kailizhishu_host']=dic['kailizhishu'][0]
    dic['kailizhishu_fair']=dic['kailizhishu'][1]
    dic['kailizhishu_guest']=dic['kailizhishu'][2]
    del dic['kailizhishu']
    return dic


dictlist=list(map(str2dict,datalist))#把datalist变成了字典形式的dictlist，即每个元素都是字典
with open('D:\\data_csv\\2014-11-30.csv','w',newline='') as f:
    w=csv.DictWriter(f,keys)
    w.writeheader()
    w.writerows(dictlist)
'''再拆成单个比赛，写入文件夹'''
df=pd.read_csv('D:\\data_csv\\2014-11-30.csv',encoding='GBK')
def bisaiquery(num):
    bisai=df.loc[lambda dfa:dfa.urlnum==num]
    return bisai

urlnumlist=list(df['urlnum'].value_counts().index)#获得比赛列表
bisailist=list(map(bisaiquery,urlnumlist))#获得由各个比赛的dataframe组成的表


def bisai2excel(bisai):#把单场比赛转换成用有多个表格的excel文件
    urlnum=str(bisai.urlnum.values[0])
    resttimelist=list(bisai.resttime.value_counts().sort_index(ascending=False).index)#获得该场比赛的变盘列表并排序
    dfdict=dict()
    for i in resttimelist:
        df=bisai.loc[lambda bisai:bisai.resttime>=i]
        dfdict[str(i)]=df.drop_duplicates('cid',keep='last')
    pn=pd.Panel(dfdict)
    pn.to_excel('D\\data_excel\\'+urlnum+'.xlsx')
    
def coprocess(bisailist):#用协程的方式并发写入
    ge = list()
    for i in bisailist:
        ge.append(gevent.spawn(bisai2excel,i))
    gevent.joinall(ge)

coprocess(bisailist)







dfb=pd.DataFrame()
resttimelist=list(dfb.resttime.value_counts().sort_index(ascending=False).index)
dfc=dfb.loc[lambda dfb:dfb.resttime>=resttimelist[1000]]#选择某个时间点的列表，比如说到第1000次更新的时刻
dfc=dfc.drop_duplicates('cid',keep='last')#只保留最大的resttime的那次更新，即为当时的状态（或许一开始的时候可以对数据集排个序，防止更新数据有误）
dfd=dfb.loc[lambda dfb:dfb.resttime>=resttimelist[600]]
dfd=dfd.drop_duplicates('cid',keep='last')#又一个数据帧
datadict={'1000':dfc,'600':dfd}#讲所有数据帧写入字典
pn=pd.Panel(datadict)#将字典转成panel
pn.to_excel('D:\\data\\pn.xlsx')#将panel写入excel，且一个数据帧作为一个sheet，此时各个表相同行都是相同的公司赔率
pnnew1=pd.read_excel('D:\\data\\pn.xlsx',sheet_name=0)#读取第一张表
pnnew1=pnnew1.dropna(axis=0,how='all')#读出表后去除空行
pnnew=pd.read_excel('D:\\data\\pn.xlsx',sheet_name=None)#读出每一张表构成一个OrderedDict
keylist=list(pnnew.keys())#获得各个表的名字，通常时以更新时间点命名的
pnnew1=pnnew[keylist[0]]#通过更新时间点的名字获得单个表

