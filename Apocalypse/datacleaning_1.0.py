#本程序是数据清洗程序用与Apocalypse 1.0 系列的开发
#每个urlnum代表一场单独的比赛，同样的urlnum下，不同的cid是同一个公司对该厂比赛的不同赔率，再同一个cid下不同的restime是不同的剩余时间
import mysql.connector
import re
import datetime
#从文本文件读入记录，由于当时写入文件的时候没分行也没分分隔符，所以整个文件只有一行
f=open('D:\\data\\20141130-20160630\\2014-11-30.txt','r')
line=f.readline()
#把这一行数据用re转成列表
datalist=re.findall('{.*?}',line)
#把列表中各个元素转成字典
for i in range(0,len(datalist)-1):
    datalist[i]=eval(datalist[i])

#把列表中各个字典去除时间戳后逐个插入mysql






mydb=mysql.connector.connect(
    host="localhost",
    user="jsy",
    passwd="921202jsy",
    database="expri"
)
mycursor=mydb.cursor()
mycursor.execute()