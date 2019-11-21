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
#再写IP的同时，再开一个去IP的进程，把每个ip的“犯规次数”做个记录，超过一定次数则去除————20191021（已完成）
#现在是通过一个全局变量permission来传递可以不可以请求的信息，或许用进程间通信会更好些————20191021（暂时不用）
#有一些UA可能是无效的，需要整理一下UA列表————20191028
#有时即便响应都是200也会出现没登录成功地情况，需要检查ajax.text的内容，如果里面有var needlogin ‘1’的字样，说明没登录上，需要重新登录————20191118
#ajax翻页的问题或许可以修改一下————20191118
#由ajax发现的登录异常也应该可以回跳回去重新登录————20191118（已解决，见下一行）
#通过修改login函数和Vorsetzen函数确保Vorsetzen结束后已经确实登录————20191121
#由于换ip需要重复的代码太多，比如login函数，可以写一个装饰器————20191121
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
import ast#用来字符串转字典


r = requests.Session()
header = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.120 Safari/537.36'}
header['Referer'] = 'http://www.okooo.com/soccer/'#必须加上这个才能进入足球日历
header['Upgrade-Insecure-Requests'] = '1'#这个也得加上
header['Proxy-Connection'] = 'keep-alive'
header['Accept'] = 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3'
header['Accept-Encoding'] = 'gzip, deflate'
header['Accept-Language'] =  'zh-CN,zh;q=0.9,en;q=0.8,de;q=0.7'
UAlist = list()
with open('D:\\data\\UAlist.txt','r') as f:
    UAlist = (f.read().splitlines())#按行读取为列表并且去掉换行符

UAlist.append('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.120 Safari/537.36')#再加一个,总共456个

def randomUA(func):#用与随机UA的装饰器
    global UAlist
    global header
    headers = header
    def decorate(*args):
        headers['User-Agent'] = random.choice(UAlist)
        return func(*args,header = headers)
    return decorate


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
    datas['AuthCode'] = ydm(filename)[1]#验证码用云打码模块识别
    return datas


@randomUA
def login(header = None):#把datas给它，它就能进行登录。应该同样也加入挂起功能
    global r
    global ippool
    error = True
    while error == True:
        try:
            ip = getip(ippool)
            r.proxies = ip[0]
            r.get('http://www.okooo.com/jingcai/',headers = header,verify=False,allow_redirects=False,timeout = 31)#从首页开启会话
            error = False
        except Exception as e:
            ip[1] += 1#加一次犯规次数
            print('Error:',e)
            print('Vorsetzen进入首页超时，正在重拨')
            error = True
    #获取验证码
    error = True
    while error == True:
        try:
            ip = getip(ippool)
            r.proxies = ip[0]
            yanzhengma = r.get('http://www.okooo.com/I/?method=ok.user.settings.authcodepic',headers = header,verify=False,allow_redirects=False,timeout = 31)#get请求登录的验证码
            error = False
        except Exception as e:
            ip[1] += 1#加一次犯规次数
            print('Error:',e)
            print('Vorsetzen获取验证码超时，正在重拨,')
            error = True
    filepath = 'D:\\data\\yanzhengma.png'
    with open(filepath,"wb") as f:
        f.write(yanzhengma.content)#保存验证码到本地
    print('已获得验证码')
    #验证码识别
    datas = randomdatas(filepath)#生成随机账户的datas
    while len(datas['AuthCode']) != 5:#如果验证码识别有问题，那就重新来
        r = requests.Session()#开启会话
        error = True
        while error == True:
            try:
                ip = getip(ippool)
                r.proxies = ip[0]#使用随机IP
                r.get('http://www.okooo.com/jingcai/',headers = header,verify=False,allow_redirects=False,timeout = 31)
                error = False
            except Exception as e:
                ip[1] += 1#加一次犯规次数
                print('Error:',e)
                print('Vorsetzen验证码识别超时，正在重拨')
                error = True               
        error = True
        while error == True:
            try:
                ip = getip(ippool)
                r.proxies = ip[0]#使用随机IP
                yanzhengma = r.get('http://www.okooo.com/I/?method=ok.user.settings.authcodepic',headers = header,verify=False,allow_redirects=False,timeout = 31)#get请求登录的验证码
                error = False
            except Exception as e:
                ip[1] += 1#加一次犯规次数
                print('Vorsetzen验证码识别超时，正在重拨4')
                error = True
        with open(filepath,"wb") as f:
            f.write(yanzhengma.content)#保存验证码到本地
        print('已重新获得验证码')
        datas = randomdatas(filepath)#生成随机账户的datas
        print('云打码已尝试一次')
    print('正在登录下面账户:')
    print(str(datas))
    error = True
    while error == True:
        try:
            ip = getip(ippool)
            r.proxies = ip[0]
            r.post('http://www.okooo.com/I/?method=ok.user.login.login',headers = header,verify=False,data = datas,allow_redirects=False,timeout = 16)#向对面服务器传送数据
            error = False
        except Exception:
            ip[1] += 1#加一次犯规次数
            print('login超时，正在重拨')
            error = True
    error = True
    while error == True:
        try:
            ip = getip(ippool)
            r.proxies = ip[0]
            r.get('http://www.okooo.com/soccer/',headers = header,verify=False,allow_redirects=False,timeout = 16)#进入足球中心
            error = False
        except Exception:
            ip[1] += 1#加一次犯规次数
            print('login超时，正在重拨')
            error = True
    header['Referer'] = 'http://www.okooo.com/soccer/'#必须加上这个才能进入足球日历
    header['Upgrade-Insecure-Requests'] = '1'#这个也得加上
    error = True
    while error == True:
        try:
            ip = getip(ippool)
            r.proxies = ip[0]
            r.get('http://www.okooo.com/soccer/match/',headers = header,verify=False,allow_redirects=False,timeout = 16)#进入足球日历,成功
            error = False
        except Exception:
            ip[1] += 1#加一次犯规次数
            print('login超时，正在重拨')
            error = True




def writeip(ippool):#每隔20秒写入ip,并为每个ip初始化犯规次数=0
    global permission
    while True:
        if len(ippool) < 5:
            permission = False
        proxycontent = requests.get('http://api.xdaili.cn/xdaili-api//privateProxy/applyStaticProxy?spiderId=0a4b8956ad274e579822b533d27f79e1&returnType=1&count=1')
        newip = re.findall('(.*?)\\r\\n',proxycontent.text)
        for j in range(0,len(newip)):
            newip[j] = [{"http":"http://" + ippool[j],},0]#为ip完善格式并初始化犯规次数
            ippool.append(newip[j])#和ippool中原本的ip列表合并
        permission = True
        time.sleep(20)#休息20秒继续


def getip(ippool):#先看看permission允不允许,直到允许，再从ippool中获取ip
    global permission
    while permission != True:
        time.sleep(5)
        continue
    ip = random.choice(ippool)
    return ip


def dropip(ippool):#当发现某个ip有问题时，从ippool中去除这个ip，并且给大家传递一个信号，等一会儿再取ip
    global permission
    while True:
        for ip in ippool:
            if ip[1] >= 5:#如果犯规次数大于5,则去除
                ippool.remove(ip)
                if len(ippool) < 5:
                    permission = False





#经试验，请求主页的ajax不需要登陆，但是请求下一周的比赛还是要登录的，所以顺序应该如常，进入主页，登陆，进入日期，获取链接，然后接下来做
def dateRange(start, end, step=1, format="%Y-%m-%d"):#生成日期列表函数，用于给datelist赋值
    strptime, strftime = datetime.strptime, datetime.strftime
    days = (strptime(end, format) - strptime(start, format)).days + 1
    return [strftime(strptime(start, format) + timedelta(i), format) for i in range(0, days, step)]

@randomUA
def jinruriqi(date,header = None):
    global r
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


@randomUA
def ajax(url,i,header = None):#从单个ajax请求的响应中获取赔率并入库
    a = r.get('http://www.okooo.com'+url+'ajax/?page='+i+'&companytype=BaijiaBooks&type=0',headers = header)
    a.encoding = 'unicode-escape'#用这个格式解码
    a.text#其中一部分即为所需要的json文件




def danchangbisai(url):#对单场比赛进行ajax请求以获得当前赔率
    ge = list()
    for i in (0,14):#这里的14好像涉及到翻页,因为最多好像就14页，但是这里或许可以改一下
        ge.append(gevent.spawn(ajax,url,i))
    gevent.joinall(ge)
    print(url)



def dangtianbisai(bisailist,date):#对列表里比赛的网址同时进行爬取
    ge = list()
    for url in bisailist:
        ge.append(gevent.spawn(danchangbisai,url))
    gevent.joinall(ge)
    print('日期'+date+'同步成功')

@randomUA
def Vorsetzen(ippool,header = None):#从打开首页到登录成功
    global r
    #先进首页
    while True:
        login()#登录账户
        wangye = r.get('http://www.okooo.com/soccer/match/?date='+'2017-01-01',headers = header,verify=False,allow_redirects=False,timeout = 9.5,proxies=random.choice(ippool))#检查是否登录成功
        if (wangye.status_code != 200) and (wangye.status_code != 203):
            print('登录失败，重新登录')
            continue
        else:
            print('登陆成功！')
            break#Vorsetzen结束
            
    


def monitoring(ippool):#总的监控程序
    while True:#无限循环地监控接下来一个月的比赛最新数据更新
        today = time.strftime("%Y-%m-%d")#今天
        nextmonth = datetime.strftime(datetime.now()+timedelta(30),"%Y-%m-%d")#下个月，威廉一些重要比赛甚至提前一个多月就出了
        datelist = dateRange(today,nextmonth)#生成日期列表
        for i in datelist:#先把过去那些天的比赛抓出来
            Vorsetzen(ippool)#每抓一天登陆一次
            bisailist = jinruriqi(i)
            dangtianbisai(bisailist,i)
     


        



ippool = list()
permission = False#设定一个允许提取ip的信号，初始为False
processwrite = Process(target=writeip,args=(ippool,))#创建写入ippool的进程
processdrop = Process(target=dropip,args=(ippool,))
processmonitor = Process(target=monitoring,args=(ippool,))
processwrite.start()
processmonitor.start()
processdrop.start()
