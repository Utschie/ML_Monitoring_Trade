#这是datacleaning的完整运行文件,先仅限于2014-11-30.txt这一个文件，测试一下速度
#bisai2excel即便是单场比赛也有5240次变盘，写入excel非常缓慢
#即便转成json，单场比赛写入后的文件居然有400M，因为拆分后每张表的keys都要重复一遍，这样就变得很大
#应该想办法把数据缩小，比如看能不能用多维数组之类的
#尝试用xarray然后以netCDF格式（.nc）存储，但是出了个问题，float object has no attribute 'encode'
#可能是因为数据集里有缺失值nan被当做字符串了，但是后面又有浮点数据，所以出错。
#只要设一个frametime作为索引，然后把所有的表连起来就可以了，这样就可以存在一个csv里————20200621
#当前的速度是一天比赛的数据预处理需要668秒，但是数据大小是2.3G，下一步是改用netCDF格式，以及去除一些无用的部分数据，看是否速度提升体积下降————20200621
#下一步是可能搞一个双索引（urlnum,frametime）,然后一天的比赛都存在一个dataframe里————20200621
#单纯的netCDF，没有sparse，并不会使体积减小。暂时只能通过删数据来缩小体积
#另外还有一种方案，就是在训练中动态处理数据并提交，可以省空间
#把gailv等于fanhuanlv除以peilv，所以把这个删掉

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
    newdict.to_csv('D:\\data_csv\\'+urlnum+'.csv')

    
    
def coprocess(bisailist):#用协程的方式并发写入
    ge = list()
    for i in bisailist:
        ge.append(gevent.spawn(bisai2csv,i))
    gevent.joinall(ge)

coprocess(bisailist)
end=time.time()
print('耗时'+str(end-start)+'秒')









