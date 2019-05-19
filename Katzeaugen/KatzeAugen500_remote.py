#实时监控澳客网赔率，经过测试，500彩票网的赔率更新与betbrains是保持一致的，而澳客网的赔率更新时间数据和500彩票网一样，所以基本上可以认为没有问题
#由于澳客网是动态加载，所以本代码从每场比赛的欧赔加载中找到原始地址,通过请求ajax的原始地址获取数据
#原始地址每个请求返回30个公司赔率数据，这样每场比赛大约6到13个请求，每周大约500到600场比赛，则最多不到8000个请求，一个月的话就是3万个请求差不多。
#如果ajax请求的服务器承受能力跟单个公司历史赔率页面相同，那么每秒50个请求来算，同步一周的比赛大约需要3分钟左右的时间
#经试验，请求主页的ajax不需要登陆，但是请求下一周的比赛还是要登录的，所以顺序应该如常，进入主页，登陆，进入日期，获取链接，然后接下来做————20190112
#ajax下来的网页解码方式是unicode-escape，与其他网页不同————20190112
#由于完整爬完某一天所有比赛的历史赔率要很长时间，所以此监控程序只监控各场比赛当前赔率，所以要想获得某一场比赛的完整赔率需要提前一个月开始监控————20190316
#先给未来一个月的比赛的网址备案，有的比赛带有赔率的把公司的名称也备案，得到未来一个月的比赛的一个数据库，也就是大约2000张表。
# 然后更新时再把这一个月的表爬一遍，如果数据有更新通过merge来更新，并把更新的数据传回到本机，本机再将新的数据和老数据叠加构成变盘数据库。————20190316
#本程序打算用多进程的方式管理ip池，一旦某个ip失效，则通过另一个进程在ip池中剔除此ip，这样防止了重复利用无效ip。如果ip数量不足，则所有请求暂时挂起，待新ip提取完毕再继续执行————20190316
#需要一个本地的mysql数据库
#为了能看到进度，现在暂时决定一天一天的爬取，而不是2000场比赛同时爬取————20190317
#因为往后一些天有的比赛就没有赔率了，应该对一些没有赔率的比赛进行筛选，这样或许可以少请求一些ajax————20190317
#在申请page=0的ajax的时候末尾会告诉你总共有多少公司，可以用这个变量决定循环次数，可以少请求几次ajax————20190317
#queue中的信息被put进去的东西只能被get一次，所以不用它来存储ippool————20190317
#permission这个变量未必有必要存在————20190317
#再dropip中如果用if的方式语句会报错那么则改用try————20190317
#login函数可以改得更简洁一些————20190317
#24小时监控更新ip池需要考虑成本问题，或者尝试其他供应商，或者限制ip池内ip总数————20190317
#开了穿梭之后需要先在cmd设置代理 set http_proxy=http://127.0.0.1:56594 set https_proxy=http://127.0.0.1:56594————20190519
#端口号56594在穿梭文件夹下的privoxy的conf里，可以自己更改————20190519
#每次循环最好重新登录一次————20190519
#先把monitoring函数的登陆写进去，然后把各个函数的随机UA加进去。然后开始写ajax入库，然后测试随机换ip策略————20190520
from gevent import monkey;monkey.patch_all()
import os
import re
import gevent
import time
import random#导入随机数模块
from bs4 import BeautifulSoup#在提取代码的时候还是要用到beautifulsoup来提取标签
from datetime import datetime, timedelta, timezone#用来把时间字符串转换成时间
import pytz#用来设置时区信息
import os#用来获取文件名列表
import requests
import urllib
import YDM
import time
import csv
import json#用来将字典写入json文件
import psutil#用来获取内存使用信息以方便释放
import copy #用来复制对象
from multiprocessing import Process,Queue#采用多进程的方式建立ip池


r = requests.Session()
header = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'}
header['Referer'] = 'http://www.okooo.com/soccer/'#必须加上这个才能进入足球日历
header['Upgrade-Insecure-Requests'] = '1'#这个也得加上
def ydm(filename):#把filepath传给它，他就能得到验证码的验证结果
    username = '921202jsy'
    password  = '921202jay'
    appid = 1
    appkey = '22cc5376925e9387a23cf797cb9ba745'
    yundama = YDM.YDMHttp(username,password,appid,appkey)
    result = yundama.decode(filename, 1005, 60)
    return result

def randomdatas(filename):#把filepath传给它，它就能得到一个随机的登录账户
    User = list()
    with open('D:\\data\\okoookonto_new.csv',"r") as f:#打开文件,并按行读取，每行为一个列表
         reader = csv.reader(f)
         for row in reader:
             User.append(row)
    datas = {
    'UserName':'',
    'PassWord':'',
    'LoginType':'okooo',
    'RememberMe':'1',
    'AuthType':'okooo',
    'AuthCode':'',
    }#datas的值取决于yundama
    suiji = random.randint(0,len(User)-1)
    datas['UserName'] = User[suiji][0]
    datas['PassWord'] = User[suiji][1]
    datas['AuthCode'] = ydm(filename)#验证码用云打码模块识别
    return datas

def login(datas):#把datas给它，它就能进行登录。应该同样也加入挂起功能
    global header
    global r
    global ippool
    header2 = header
    error = True
    while error == True:
        try:
            r.post('http://www.okooo.com/I/?method=ok.user.login.login',headers = header2,verify=False,data = datas,allow_redirects=False,timeout = 16)#向对面服务器传送数据
            error = False
        except Exception:
            print('login超时，正在重拨')
            r.proxies = random.choice(ippool)#换一个ip
            error = True
    error = True
    while error == True:
        try:
            r.get('http://www.okooo.com/soccer/',headers = header2,verify=False,allow_redirects=False,timeout = 16)#进入足球中心
            error = False
        except Exception:
            print('login超时，正在重拨')
            r.proxies = random.choice(ippool)#换一个ip
            error = True
    header2['Referer'] = 'http://www.okooo.com/soccer/'#必须加上这个才能进入足球日历
    header2['Upgrade-Insecure-Requests'] = '1'#这个也得加上
    error = True
    while error == True:
        try:
            r.get('http://www.okooo.com/soccer/match/',headers = header2,verify=False,allow_redirects=False,timeout = 16)#进入足球日历,成功
            error = False
        except Exception:
            print('login超时，正在重拨')
            r.proxies = random.choice(ippool)#换一个ip
            error = True





def writeip(ippool):#每隔20秒写入ip
    global permission
    while True:
        if len(ippool) < 5:
            permission = False
        proxycontent = requests.get('http://api.xdaili.cn/xdaili-api//privateProxy/applyStaticProxy?spiderId=0a4b8956ad274e579822b533d27f79e1&returnType=1&count=1')
        proxylist = re.findall('(.*?)\\r\\n',proxycontent.text)
        for j in range(0,len(proxylist)):
            proxylist[j] = {"http":"http://" + proxylist[j],}#为ip完善格式
            ippool = ippool+proxylist[j]#和ippool中原本的ip列表合并
        permission = True
        time.sleep(20)#休息20秒继续


def getip(ippool,q):#先看看permission允不允许,直到允许，再从ippool中获取ip
    global permission
    while permission != True:
        continue
    proxy = random.choice(ippool)
    return proxy


def dropip(ip,ippool):#当发现某个ip有问题时，从ippool中去除这个ip，并且给大家传递一个信号，等一会儿再取ip
    global permission
    if ip in ippool:#如果无效ip在池中，则挂起其他程序并去除它
        permission = False#此函数运行时所有请求挂起
        ippool.remove(ip)
        if len(ippool) < 5:
            permission = False
        permission = True#去掉无效ip后所有请求再执行





#经试验，请求主页的ajax不需要登陆，但是请求下一周的比赛还是要登录的，所以顺序应该如常，进入主页，登陆，进入日期，获取链接，然后接下来做
def dateRange(start, end, step=1, format="%Y-%m-%d"):#生成日期列表函数，用于给datelist赋值
    strptime, strftime = datetime.strptime, datetime.strftime
    days = (strptime(end, format) - strptime(start, format)).days + 1
    return [strftime(strptime(start, format) + timedelta(i), format) for i in range(0, days, step)]


def jinruriqi(date):
    global r
    global header
    global ippool
    wangye = r.get('http://www.okooo.com/soccer/match/?date='+date,headers = header,verify=False,allow_redirects=False,timeout = 9.5,proxies=random.choice(ippool))
    content1 = wangye.content.decode('gb18030')#取出wangye的源代码
    sucker1 = '/soccer/match/.*?/odds/'
    bisaiurl = re.findall(sucker1,content1)#获得当天的比赛列表
    return bisaiurl
#def riqiliebiao():
#    today = time.strftime("%Y-%m-%d")#今天
#    nextmonth = datetime.strftime(datetime.now()+timedelta(35),"%Y-%m-%d")#下个月，威廉一些重要比赛甚至提前一个多月就出了
#    datelist = dateRange(today,nextmonth)#生成日期列表
#    bisailist = list()
#    for i in datelist:#获得一个月内所有比赛的url
#        bisailist = bisailist + jinruriqi(i)
#    return bisailist
def ajax(url,i):#从单个ajax请求的响应中获取赔率并入库
    global header
    a = r.get(url+'http://www.okooo.com/soccer/match/1052796/odds/ajax/?page=1&companytype=BaijiaBooks&type=0',headers = header)
    a.encoding = 'unicode-escape'#用这个格式解码
    a.text#其中一部分即为所需要的json文件


def danchangbisai(url):#对单场比赛进行ajax请求以获得当前赔率
    ge = list()
    for i in (0,14):
        ge.append(gevent.spawn(ajax,url,i))
    gevent.joinall(ge)
    print(url)



def dangtianbisai(bisailist,date):#对列表里比赛的网址同时进行爬取
    ge = list()
    for i in bisailist:
        ge.append(gevent.spawn(danchangbisai,i))
    gevent.joinall(ge)
    print('日期'+date+'同步成功')


def monitoring(ippool):#总的监控程序
    while True:#无限循环
        today = time.strftime("%Y-%m-%d")#今天
        nextmonth = datetime.strftime(datetime.now()+timedelta(35),"%Y-%m-%d")#下个月，威廉一些重要比赛甚至提前一个多月就出了
        datelist = dateRange(today,nextmonth)#生成日期列表
        for i in datelist:
            bisailist = jinruriqi(i)
            dangtianbisai(bisailist,i)
        print('同步已完成')



ippool = list()
permission = False#设定一个允许提取ip的信号，初始为False
pw = Process(target=writeip,args=(ippool,))#创建写入ippool的进程
pm = Process(target=monitoring,args=(ippool,))
pw.start()
pm.start()
