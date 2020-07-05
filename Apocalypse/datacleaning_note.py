'''本程序是数据清洗程序用与Apocalypse 1.0 系列的开发的笔记本
除了记录了清洗过程还记录了后续数据处理中的读取方法
'''


#每个urlnum代表一场单独的比赛，同样的urlnum下，不同的cid是同一个公司对该厂比赛的不同赔率，再同一个cid下不同的restime是不同的剩余时间
#用sql很麻烦，改用pandas做成数据框，然后直接洗试试
#现在是把字典列表逐个插入dataframe，写进一天的文件要好多分钟————20200618
#一个76.5M的原文件写入dataframe要5472秒，压缩成csv变成28.6M，读取只需1秒，所以要先把所有的文件转成csv文件————20200618
#按上面那个速度400G数据全转完需要8000多个小时，需要用并行的方式插入数据框————20200618（已解决）
#找到问题了，其实数据转换很快，但是插入pandas的dataframe的过程特别慢，那就直接写入csv吧
#直接写入csv用时18秒，非常棒
#resttime是以分钟为单位的
#最终是要以每天做一个文件夹，每个文件夹里带有着以urlnum命名的当天比赛的名字的文件，每个文件里存放着密度棒序列————20200620
#
import re
import datetime
import pandas as pd
import time
import csv
#从文本文件读入记录，由于当时写入文件的时候没分行也没分分隔符，所以整个文件只有一行
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

def dict2DataFrame(dic):
    global df
    df=df.append(dic,ignore_index=True)

start=time.time()
dictlist=list(map(str2dict,datalist))#把datalist变成了字典形式的dictlist，即每个元素都是字典
with open('D:\\data_csv\\2014-11-30.csv','w',newline='') as f:
    w=csv.DictWriter(f,keys)
    w.writeheader()
    w.writerows(dictlist)

df=pd.read_csv('D:\\data_csv\\2014-11-30.csv',encoding='GBK')#再讲csv读入dataframe，注意编码
#接下来进行数据重组处理
dfa=df.copy()#复制dfa用来做试验
a=df['urlnum'].value_counts()#获得urlnum的所有值
bisailist=list(a.index)#把urlnum的所有值转换成一个列表，可以用来循环处理
dfb=dfa.loc[lambda dfa:dfa.urlnum==681079]#用调用callable的方式选择其中一场比赛（比如第一场）
resttimelist=list(dfb.resttime.value_counts().sort_index(ascending=False).index)#得到单场比赛的不同变盘时间点作为一个列表并从早到晚排序，共变了2702次
#下面按着剩余时间对数据切片，产生2702个切片，不知道是这样好一点还是保留时间戳处理好一点
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








end=time.time()
print('耗时'+str(end-start)+'秒')







