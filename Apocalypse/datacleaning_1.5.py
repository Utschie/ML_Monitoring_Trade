#此文件是datacleaning_1.5，比1.0提升部分性能
#删除了冗余数据，只保留与最简单的模型相关的数据后，时间变成470秒，数据体积767MB————20200621
from gevent import monkey;monkey.patch_all()
import gevent
import re
import datetime
import pandas as pd
import time
import csv
start=time.time()
'''先转成csv'''
#进入文件夹，生成文件名序列
f=open('D:\\data\\20141130-20160630\\2014-11-30.txt','r')
line=f.readline()
f.close()#要关闭数据集
#把这一行数据用re转成列表
datalist=re.findall('{.*?}',line)
keys=['date', 'urlnum', 'cid', 'resttime', 'fanhuanlv', 
'peilv_host', 'peilv_fair', 'peilv_guest', 
'kailizhishu_host', 'kailizhishu_fair', 'kailizhishu_guest']
df=pd.DataFrame(columns=keys)#先列好字段
#把列表中各个元素转成字典,并且把peilv，gailv和kailizhishu分拆成三列，否则无法正确读入pandas
def str2dict(str):#讲datalist中的单个元素转换插入dataframe的函数
    dic=eval(str)
    del dic['timestamp']
    del dic['zhudui']
    del dic['kedui']
    del dic['companyname']
    del dic['league']
    dic['peilv_host']=dic['peilv'][0]
    dic['peilv_fair']=dic['peilv'][1]
    dic['peilv_guest']=dic['peilv'][2]
    del dic['peilv']
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
def bisai2csv(bisai):#把单场比赛转换成csv文件
    urlnum=str(bisai.urlnum.values[0])
    resttimelist=list(bisai.resttime.value_counts().sort_index(ascending=False).index)#获得该场比赛的变盘列表并排序
    dfdict=list()
    for i in resttimelist:
        df=bisai.loc[lambda bisai:bisai.resttime>=i]
        df['frametime']=i
        df=df.set_index('frametime')
        dfdict.append(df.drop_duplicates('cid',keep='last'))
    newdict=pd.concat(dfdict)#一个新的
    newdict=newdict.drop(columns=['resttime','urlnum','date'])
    newdict.to_csv('D:\\data_csv\\'+urlnum+'.csv')

    
    
def coprocess(bisailist):#用协程的方式并发写入
    ge = list()
    for i in bisailist:
        ge.append(gevent.spawn(bisai2csv,i))
    gevent.joinall(ge)

coprocess(bisailist)
end=time.time()
print('耗时'+str(end-start)+'秒')

