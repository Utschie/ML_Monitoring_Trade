#本文件用来洗20160701-20190224的数据，装在4TB的盘里
#首先手动在4TB盘内建立一个名为data_csv_new_20160701-20190224的文件夹————20201128
#应该改成日期池，然后进程在日期里抽日期而不是固定的分成多少天，这样能充分利用电脑性能————20201202
#20160701-20190224总共有66709场比赛，20141130-20160630共有37897场比赛，总共104606场比赛————20201203
#如果再把20190204-20201231的数据拿下来，那么预计还会增加45000+场比赛，当前的数据量应该也差不多够用了，反正是一个三分类问题————20201203
from gevent import monkey;monkey.patch_all()
import gevent
import re
import datetime
import pandas as pd
import time
import csv
from multiprocessing import Process
import os#用与建立文件夹
import numpy as np
'''先转成csv'''
#进入文件夹，生成文件名序列
with open('D:\\data\\cidlist.csv') as f:
    reader = csv.reader(f)
    cidlist = [row[1] for row in reader]#得到cid对应表

cidlist = list(map(float,cidlist))#把各个元素字符串类型转成浮点数类型
'''先转成csv'''
#进入文件夹，生成文件名序列
def txt2csv(date):#把原始的按天分的txt文件转成按天分的csv文件
    filepath='G:\\okooofile_20160701-20190224\\'+date+'.txt'#讲日期转成文件名
    f=open(filepath,'r')
    line=f.readline()
    f.close()#要关闭数据集
    #把这一行数据用re转成列表
    datalist=re.findall('{.*?}',line)
    keys=['date', 'urlnum', 'league', 'cid', 'zhudui', 'kedui', 'companyname', 'resttime', 'fanhuanlv', 
    'peilv_host', 'peilv_fair', 'peilv_guest', 
    'gailv_host', 'gailv_fair', 'gailv_guest', 
    'kailizhishu_host', 'kailizhishu_fair', 'kailizhishu_guest']
    df=pd.DataFrame(columns=keys)#先列好字段,一个空的dataframe，然后等下面从csv中读取是再
    dictlist=list(map(str2dict,datalist))#把datalist变成了字典形式的dictlist，即每个元素都是字典
    with open('H:\\data_csv_new_20160701-20190224\\'+date+'.csv','w',newline='') as f:#
        w=csv.DictWriter(f,keys)
        w.writeheader()
        w.writerows(dictlist)
    df=pd.read_csv('H:\\data_csv_new_20160701-20190224\\'+date+'.csv',encoding='GBK')#在移动硬盘内写入同名的csv文件并读取
    return df

#把列表中各个元素转成字典,并且把peilv，gailv和kailizhishu分拆成三列，否则无法正确读入pandas
def str2dict(string):#讲datalist中的单个元素转换插入dataframe的函数,跟之前不同的是除了timestamp被删除，其余全部保留
    dic=eval(string)
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


def bisaiquery(df):#因为后面的map函数只能接受一个参数的列表，所以在此嵌套一下，然后返回相应的函数
    def bisaiquery_in(num):
        bisai=df.loc[lambda dfa:dfa.urlnum==num]
        return bisai
    return bisaiquery_in

    

def bisai2csv(bisai):#把单场比赛转换成csv文件
    #############准备好csv文件内容###########
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
    outputpath1='H:\\cleaned_data_new_20160701-20190224\\'+date+'\\'+urlnum+'.csv'
    #outputpath2 = 'F:\\cleaned_data_new_dflist_20141130-20160630\\'+date+'\\'+urlnum+'.npz'
    newdict.to_csv(outputpath1)#输出csv文件
    ##################准备好npz文件内容###########
    '''
    data = pd.read_csv(outputpath1)#读取文件
    data = data.drop(columns=['league','zhudui','kedui','companyname'])#去除非数字的列
    frametimelist=data.frametime.value_counts().sort_index(ascending=False).index#将frametime的值读取成列表
    framelist = list()#framelist为一个空列表
    for i in frametimelist:#其中frametimelist里的数据是整型
        state = data.groupby('frametime').get_group(i)#从第一次变盘开始得到当次转移
        state = np.array(state)#转成numpy多维数组
        #在填充成矩阵之前需要知道所有数据中到底有多少个cid
        statematrix=np.zeros((410,12))#
        for j in state:
            cid = j[1]#得到浮点数类型的cid
            index = cidlist.index(cid)
            statematrix[index] = j#把对应矩阵那一行给它
        statematrix=np.delete(statematrix,(0,1), axis=-1)#去掉frametime和cid列
        framelist.append(statematrix)
    framelist = np.array(framelist)#转成numpy数组
    frametimelist = np.array(frametimelist)
    np.savez(outputpath2,framelist=framelist,frametimelist=frametimelist)#framelist和frametimelist分别是自定义的key，将来读取用这两个key来引用
    '''
   
    
def coprocess(bisailist):#用协程的方式并发写入
    ge = list()
    for i in bisailist:
        ge.append(gevent.spawn(bisai2csv,i))
    gevent.joinall(ge)


def proc(datelist):
    for i in datelist:
        start=time.time()
        outputpath1='H:\\cleaned_data_new_20160701-20190224\\'+i#为这一天建立一个文件夹
        #outputpath2='F:\\cleaned_data_new_dflist_20141130-20160630\\'+i
        os.makedirs(outputpath1)#建立保存csv的文件夹
        #os.makedirs(outputpath2)#建立保存npz的文件夹
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
    filelist = os.listdir('G:\\okooofile_20160701-20190224')#读出这一年半的数据文件名
    datelist=[i[0:-4] for i in filelist]
    datelist_list=listdivision(datelist,243)
    process_list = []
    for i in datelist_list:
        p = Process(target=proc,args=(i,))
        p.start()
        process_list.append(p)


    for i in process_list:
        p.join()









