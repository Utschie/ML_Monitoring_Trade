#此文件是datacleaning_1.5，比1.0提升部分性能
#删除了冗余数据，只保留与最简单的模型相关的数据后，时间变成470秒，数据体积767MB————20200621
#最原始的txt数据大小是76.5M,转换后变成767M，数据体积扩大了10倍。还算可以接受的范围
#暂时还没有测试————20200624
#最好还加一个多进程，开3个核跑,cpu利用率大约50%
#把文件拷入F盘，然后conda里切换盘符到F盘，执行python datacleaning_1.5.py
#在清洗数据是出现了一个问题，就是memory_error，应该建立一个写入log文件，来查看可能出现的错误————20200626
#另外线程或许应该可以公共处理同一个列表，来保证不会因为某一天比赛的memory错误导致后面的几十天比赛都无法清洗————20200626
#问题出在2015-08-08这天单天的数据竟高达1.6G，然后内存超了虚拟内存又不足，把E盘划出一部分给虚拟内存就好了————20200627
#还要把中间空余的16年上半年数据洗了
#20150501的数据好像有点小了，可以重新洗一下
from gevent import monkey;monkey.patch_all()
import gevent
import re
import datetime
import pandas as pd
import time
import csv
from multiprocessing import Process
import os#用与建立文件夹
'''先转成csv'''
#进入文件夹，生成文件名序列
def txt2csv(date):
    filepath='D:\\data\\20141130-20160630\\'+date+'.txt'#讲日期转成文件名
    f=open(filepath,'r')
    line=f.readline()
    f.close()#要关闭数据集
    #把这一行数据用re转成列表
    datalist=re.findall('{.*?}',line)
    keys=['date', 'urlnum', 'cid', 'resttime', 'fanhuanlv', 
    'peilv_host', 'peilv_fair', 'peilv_guest', 
    'kailizhishu_host', 'kailizhishu_fair', 'kailizhishu_guest']
    df=pd.DataFrame(columns=keys)#先列好字段
    dictlist=list(map(str2dict,datalist))#把datalist变成了字典形式的dictlist，即每个元素都是字典
    with open('F:\\data_csv\\'+date+'.csv','w',newline='') as f:#
        w=csv.DictWriter(f,keys)
        w.writeheader()
        w.writerows(dictlist)
    df=pd.read_csv('F:\\data_csv\\'+date+'.csv',encoding='GBK')#在移动硬盘内写入同名的csv文件并读取
    return df

#把列表中各个元素转成字典,并且把peilv，gailv和kailizhishu分拆成三列，否则无法正确读入pandas
def str2dict(string):#讲datalist中的单个元素转换插入dataframe的函数
    dic=eval(string)
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


def bisaiquery(df):#因为后面的map函数只能接受一个参数的列表，所以在此嵌套一下，然后返回相应的函数
    def bisaiquery_in(num):
        bisai=df.loc[lambda dfa:dfa.urlnum==num]
        return bisai
    return bisaiquery_in

    

def bisai2csv(bisai):#把单场比赛转换成csv文件
    urlnum=str(bisai.urlnum.values[0])
    date=str(bisai.date.values[0])
    resttimelist=list(bisai.resttime.value_counts().sort_index(ascending=False).index)#获得该场比赛的变盘列表并排序
    dfdict=list()
    for i in resttimelist:
        df=bisai.loc[lambda bisai:bisai.resttime>=i]
        df['frametime']=i
        df=df.set_index('frametime')
        dfdict.append(df.drop_duplicates('cid',keep='last'))
    newdict=pd.concat(dfdict)#一个新的
    newdict=newdict.drop(columns=['resttime','urlnum','date'])
    outputpath='F:\\cleaned_data_20141130-20160630\\'+date+'\\'+urlnum+'.csv'
    newdict.to_csv(outputpath)

    
    
def coprocess(bisailist):#用协程的方式并发写入
    ge = list()
    for i in bisailist:
        ge.append(gevent.spawn(bisai2csv,i))
    gevent.joinall(ge)


def proc(datelist):
    for i in datelist:
        start=time.time()
        outputpath='F:\\cleaned_data_20141130-20160630\\'+i#为这一天建立一个文件夹
        os.makedirs(outputpath)#建立文件夹
        df=txt2csv(i)#将txt文件导出csv后读入dataframe
        urlnumlist=list(df['urlnum'].value_counts().index)#获得当天比赛列表
        bisailist=list(map(bisaiquery(df),urlnumlist))#获得由各个比赛的dataframe组成的表
        coprocess(bisailist)#协程并发分别写入文件
        end=time.time()
        outputstr='日期：'+i+'已清洗完成\n'+'耗时：'+str(end-start)+'秒\n'
        print(outputstr)


def listdivision(listTemp, n):
    list_list=list()
    for i in range(0, len(listTemp), n):
        list_list.append(listTemp[i:i + n])
    return list_list

    

if __name__ == '__main__':
    filelist = os.listdir('D:\\data\\20141130-20160630')#读出这一年半的数据文件名
    datelist=[i[0:-4] for i in filelist]
    datelist_list=listdivision(datelist,146)
    process_list = []
    for i in datelist_list:
        p = Process(target=proc,args=(i,))
        p.start()
        process_list.append(p)


    for i in process_list:
        p.join()









