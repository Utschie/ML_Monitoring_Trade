#此程序是用来调试个别函数的程序，让每次调试都在最短时间做好准备工作
#先把代码全都加载
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

ippool = list()
UAlist = list()
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
def login(datas,header = None):#把datas给它，它就能进行登录。应该同样也加入挂起功能
    global r
    global ippool
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
    a = r.get(url+'http://www.okooo.com/soccer/match/1052796/odds/ajax/?page='+i+'&companytype=BaijiaBooks&type=0',headers = header)
    a.encoding = 'unicode-escape'#用这个格式解码
    a.text#其中一部分即为所需要的json文件



def danchangbisai(url):#对单场比赛进行ajax请求以获得当前赔率
    ge = list()
    for i in (0,14):#这里的14好像涉及到翻页
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
    login(datas)#登录账户
    print('正在登录下面账户:')
    print(str(datas))
    print('登陆成功！')


def monitoring(ippool):#总的监控程序
    while True:#无限循环地监控接下来一个月的比赛最新数据更新
        today = time.strftime("%Y-%m-%d")#今天
        nextmonth = datetime.strftime(datetime.now()+timedelta(30),"%Y-%m-%d")#下个月，威廉一些重要比赛甚至提前一个多月就出了
        datelist = dateRange(today,nextmonth)#生成日期列表
        for i in datelist:#先把过去那些天的比赛抓出来
            Vorsetzen(ippool)#每抓一天登陆一次
            bisailist = jinruriqi(i)
            dangtianbisai(bisailist,i)



##########################################################下面是开始一步一步进入###########################################
today = time.strftime("%Y-%m-%d")#今天
nextmonth = datetime.strftime(datetime.now()+timedelta(30),"%Y-%m-%d")#下个月，威廉一些重要比赛甚至提前一个多月就出了
datelist = dateRange(today,nextmonth)#生成日期列表
r = requests.Session()
header = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.120 Safari/537.36'}
header['Host'] = 'www.okooo.com'#必须加上这个才能进入足球日历
header['Upgrade-Insecure-Requests'] = '1'#这个也得加上
r.get('http://www.okooo.com/jingcai/',headers = header,verify=False,allow_redirects=False,timeout = 31)#从首页开启会话
yanzhengma = r.get('http://www.okooo.com/I/?method=ok.user.settings.authcodepic',headers = header,verify=False,allow_redirects=False,timeout = 31)
filepath = 'D:\\data\\yanzhengma.png'
with open(filepath,"wb") as f:
    f.write(yanzhengma.content)#保存验证码到本地


#验证码识别
datas = randomdatas(filepath)#生成随机账户的datas
r.post('http://www.okooo.com/I/?method=ok.user.login.login',headers = header,verify=False,data = datas,allow_redirects=False,timeout = 16)#登陆
r.get('http://www.okooo.com/soccer/',headers = header,verify=False,allow_redirects=False,timeout = 16)#进入足球中心
r.get('http://www.okooo.com/soccer/match/',headers = header,verify=False,allow_redirects=False,timeout = 16)#进入足球日历
wangye = r.get('http://www.okooo.com/soccer/match/?date='+datelist[1],headers = header,verify=False,allow_redirects=False,timeout = 9.5)#进入指定日期
content1 = wangye.content.decode('gb18030')#取出wangye的源代码
sucker1 = '/soccer/match/.*?/odds/'
bisaiurl = re.findall(sucker1,content1)#获得当天的比赛列表
url = bisaiurl[1]
#在ajax请求之前，一直到已经获得了未来的某一天都不需要登录，直到请求未来某一天的其中一场比赛时才要求登录
ajax = r.get('http://www.okooo.com'+url+'ajax/?page='+'1'+'&companytype=BaijiaBooks&type=0',headers = header)#请求当天某一场比赛的ajax
ajax.encoding = 'unicode-escape'#用这个格式解码
text = ajax.text#其中一部分即为所需要的json文件

#下面开始提取数据
sucker_ajax = 'var data_str = \'\[(.*)]\''
peilv_str = re.search(sucker_ajax,text).group(1)#这样就得到了装有data_str的字符串
peilv_dict = ast.literal_eval(peilv_str)#这样就得到了长度为30的tuple，每个元素是一个公司赔率，为字典的形式
#接下来把字典写入mysql数据库