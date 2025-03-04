#本程序是为了补充爬取各场比赛的赛果的
#他妈的云打码好像黄了
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
import time
import csv
import json#用来将字典写入json文件
import psutil#用来获取内存使用信息以方便释放
import copy #用来复制对象
import base64
from io import BytesIO
from PIL import Image
from sys import version_info



def checkip(ip):
    global header
    global UAlist
    header4 = header
    iplist = ip
    for i in range(0,len(iplist)):
        error4 = True
        mal3 = 1
        while (error4 ==True and mal3 <= 3):#总共拨三次，首拨1次重拨2次
            try:
                header4['User-Agent'] = random.choice(UAlist)#每尝试一次换一次UA
                check = requests.get('http://www.okooo.com/jingcai/',headers = header4,proxies = {"http":"http://"+ iplist[i]},timeout = 6.5)
            except Exception as e:
                error4 = True
                mal3 = mal3 + 1
                if mal3 > 3:
                    iplist[i] = ''
                    print('第' + str(i) + '个IP不合格，已去除')
            else:
                error4 = False
                print('第' + str(i) + '个IP合格')
    while '' in iplist:
        iplist.remove('')
    return iplist


def dateRange(start, end, step=1, format="%Y-%m-%d"):#生成日期列表函数，用于给datelist赋值
    strptime, strftime = datetime.strptime, datetime.strftime
    days = (strptime(end, format) - strptime(start, format)).days + 1
    return [strftime(strptime(start, format) + timedelta(i), format) for i in range(0, days, step)]


def ydm(uname, pwd,  img):
    img = img.convert('RGB')
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    if version_info.major >= 3:
        b64 = str(base64.b64encode(buffered.getvalue()), encoding='utf-8')
    else:
        b64 = str(base64.b64encode(buffered.getvalue()))
    data = {"username": uname, "password": pwd, "image": b64}
    result = json.loads(requests.post("http://api.ttshitu.com/base64", json=data).text)
    if result['success']:
        return result["data"]["result"]
    else:
        return result["message"]
    return ""


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
    datas['AuthCode'] = ydm('921202jsy','921202jay',Image.open(filename))#验证码用云打码模块识别
    return datas

def login(datas):#把datas给它，它就能进行登录,不切换ip
    global header
    global r
    global proxylist
    header2 = header
    error = True
    while error == True:
        try:
            denglu = r.post('http://www.okooo.com/I/?method=ok.user.login.login',headers = header2,verify=False,data = datas,allow_redirects=False,timeout = 16)#向对面服务器传送数据
            error = False
        except Exception as e:
            print('login超时，正在重拨')
            r.proxies = random.choice(proxylist)#换一个ip
            error = True
    error = True
    while error == True:
        try:
            zuqiuzhongxin = r.get('http://www.okooo.com/soccer/',headers = header2,verify=False,allow_redirects=False,timeout = 16)#进入足球中心
            error = False
        except Exception as e:
            print('login超时，正在重拨')
            r.proxies = random.choice(proxylist)#换一个ip
            error = True
    header2['Referer'] = 'http://www.okooo.com/soccer/'#必须加上这个才能进入足球日历
    header2['Upgrade-Insecure-Requests'] = '1'#这个也得加上
    error = True
    while error == True:
        try:
            zuqiurili = r.get('http://www.okooo.com/soccer/match/',headers = header2,verify=False,allow_redirects=False,timeout = 16)#进入足球日历,成功
            error = False
        except Exception as e:
            print('login超时，正在重拨')
            r.proxies = random.choice(proxylist)#换一个ip
            error = True

class Startpoint(object):#定义起始点类，给出日志路径就能得到爬去日期和比赛场次
    def __init__(self,logpath):
        self.logpath = logpath
        log = open(self.logpath,'r')
        try:
            logrecord = log.readline().strip('\n')
            log.close()
            if logrecord != '':
                self.startdate = logrecord[0:10]#前十位是日期
                self.startgame = int(logrecord[10:])#后面是比赛的序号
            else:
                self.startdate = datetime.now().strftime('%Y-%m-%d')
                self.startgame = '0'
        except Exception as e:
            print('Error:',e)
            self.startdate = datetime.now().strftime('%Y-%m-%d')
            self.startgame = '0'

def dangtianbisai(date):
    global header
    global r
    global proxylist
    global UAlist
    starttime = time.time()
    header3 = header
    header3['Referer'] = 'http://www.okooo.com/soccer/'#必须加上这个才能进入足球日历
    header3['Upgrade-Insecure-Requests'] = '1'#这个也得加上
    header3['User-Agent'] = random.choice(UAlist)
    error = True
    while error == True:
        try:
            wangye = r.get('http://www.okooo.com/soccer/match/?date=' + date,headers = header3,verify=False,allow_redirects=False,timeout = 31)
            error = False
        except Exception as e:
            print('dangtianbisai超时1，10秒后重拨')
            header3['User-Agent'] = random.choice(UAlist)#出错了才换UA
            r.proxies = random.choice(proxylist)#出错了才换IP
            time.sleep(10)
            error = True
    print('进入日期：'+ date)
    content1 = wangye.content.decode('gb18030','ignore')#取出wangye的源代码,忽略无法解码的部分
    sucker1 = '/soccer/match/.*?/odds/'
    bisaiurl = re.findall(sucker1,content1)#获得当天的比赛列表
    sucker2 = 'href="/soccer/match/(.*?)/" target="_blank">(.*?)-(.*?)</a></b></td>'
    bisai_results = re.findall(sucker2,content1)#得到比赛和比分列表，列表元素为一个三个元素的元组
    return bisai_results











def main():#从打开首页到登录成功
    global header
    global r
    global proxylist
    error = True
    while error == True:
        try:
            r.get('http://www.okooo.com/jingcai/',headers = header,verify=False,allow_redirects=False,timeout = 31)#从首页开启会话
            error = False
        except Exception as e:
            print('Error:',e)
            print('main超时，正在重拨1')
            r.proxies = random.choice(proxylist)
            error = True
    error = True
    mal3 = 0
    while error == True:
        try:
            yanzhengma = r.get('http://www.okooo.com/I/?method=ok.user.settings.authcodepic',headers = header,verify=False,allow_redirects=False,timeout = 31)#get请求登录的验证码
            error = False
        except Exception as e:
            if (mal3%3 != 0 or mal3 == 0):
                mal3 = mal3 + 1
                print('Error:',e)
                print('main超时，正在进行第'+str(mal3)+'次重拨2,')
                r.proxies = random.choice(proxylist)
                error = True
            else:
                print('main获取验证码失败，10秒后重启回话，重新提取ip')
                r.close()
                time.sleep(10)
                header = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'}#设置UA假装是浏览器
                header['User-Agent'] = random.choice(UAlist)
                proxycontent = requests.get('http://api.xdaili.cn/xdaili-api//privateProxy/applyStaticProxy?spiderId=0a4b8956ad274e579822b533d27f79e1&returnType=1&count=1') #接入混拨代理
                print('已获取IP')
                proxylist = re.findall('(.*?)\\r\\n',proxycontent.text)
                print('正在检查IP')
                proxylist = checkip(proxylist)
                for j in range(0,len(proxylist)):
                    proxylist[j] = {"http":"http://" + proxylist[j],}
                print(proxylist)
                while (len(proxylist) <=4):
                    print('有效ip数目不足，需等待15秒重新提取')
                    time.sleep(10)
                    proxycontent = requests.get('http://api.xdaili.cn/xdaili-api//privateProxy/applyStaticProxy?spiderId=0a4b8956ad274e579822b533d27f79e1&returnType=1&count=1')
                    print('已获取IP')
                    proxylist = re.findall('(.*?)\\r\\n',proxycontent.text)
                    print('正在检查IP')
                    proxylist = checkip(proxylist)
                    for j in range(0,len(proxylist)):
                        proxylist[j] = {"http":"http://" + proxylist[j],}
                    print(proxylist)
                r = requests.Session()#开启会话
                r.proxies = random.choice(proxylist)
                error = True           
    filepath = 'D:\\data\\yanzhengma.png'
    with open(filepath,"wb") as f:
        f.write(yanzhengma.content)#保存验证码到本地
    print('已获得验证码')
    datas = randomdatas(filepath)#生成随机账户的datas
    while len(datas['AuthCode']) != 5:#如果验证码识别有问题，那就重新来
        r = requests.Session()#开启会话
        r.proxies = random.choice(proxylist)#使用随机IP
        error = True
        while error == True:
            try:
                r.get('http://www.okooo.com/jingcai/',headers = header,verify=False,allow_redirects=False,timeout = 31)
                error = False
            except Exception as e:
                print('Error:',e)
                print('main超时，正在重拨3')
                r.proxies = random.choice(proxylist)
                error = True
        error = True
        while error == True:
            try:
                yanzhengma = r.get('http://www.okooo.com/I/?method=ok.user.settings.authcodepic',headers = header,verify=False,allow_redirects=False,timeout = 31)#get请求登录的验证码
                error = False
            except Exception as e:
                print('main超时，正在重拨4')
                r.proxies = random.choice(proxylist)
                error = True
        with open(filepath,"wb") as f:
            f.write(yanzhengma.content)#保存验证码到本地
        print('已重新获得验证码')
        datas = randomdatas(filepath)#生成随机账户的datas
        print('云打码已尝试一次')
    login(datas)#登录账户
    print('正在登录下面账户:')
    print(str(datas))





start = time.time()
UAcontent = urllib.request.urlopen('file:///D:/data/useragentswitcher.xml').read()
UAcontent = str(UAcontent)
UAname = re.findall('(useragent=")(.*?)(")',UAcontent)
UAlist = list()
for z in range(0,int(len(UAname))):
    UAlist.append(UAname[z][1])

UAlist = UAlist[0:586]#这样就得到了一个拥有586个UA的UA池
UAlist.append('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36')#再加一个
logpath = 'D:\\data\\okooolog.txt'
error = True
n = 0
while error == True:
    beginpoint = Startpoint(logpath)#得到起始点信息
    datelist = dateRange("2016-07-01", beginpoint.startdate)#生成一个到起始点信息的日期列表
    datelist.reverse()#让列表倒序，使得爬虫从最近的一天往前爬
    try:
        for i in datelist:#开启一个循环，保证爬取每天的数据用的UA，IP，账户都不一样
            header = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'}#设置UA假装是浏览器
            header['User-Agent'] = random.choice(UAlist)
            proxycontent = requests.get('http://api.xdaili.cn/xdaili-api//privateProxy/applyStaticProxy?spiderId=0a4b8956ad274e579822b533d27f79e1&returnType=1&count=1') #接入混拨代理
            print('已获取IP')
            proxylist = re.findall('(.*?)\\r\\n',proxycontent.text)
            print('正在检查IP')
            proxylist = checkip(proxylist)
            for j in range(0,len(proxylist)):
                proxylist[j] = {"http":"http://" + proxylist[j],}
            print(proxylist)
            while (len(proxylist) <=4):
                print('有效ip数目不足，需等待15秒重新提取')
                time.sleep(10)
                proxycontent = requests.get('http://api.xdaili.cn/xdaili-api//privateProxy/applyStaticProxy?spiderId=0a4b8956ad274e579822b533d27f79e1&returnType=1&count=1')
                print('已获取IP')
                proxylist = re.findall('(.*?)\\r\\n',proxycontent.text)
                print('正在检查IP')
                proxylist = checkip(proxylist)
                for j in range(0,len(proxylist)):
                    proxylist[j] = {"http":"http://" + proxylist[j],}
                print(proxylist)
            r = requests.Session()#开启会话
            r.proxies = random.choice(proxylist)
            main()
            ceshi = r.get('http://www.okooo.com/soccer/match/?date=2017-01-01',headers = header,verify=False,allow_redirects=False,timeout = 31)#进入1月1日，看看有没有重定向，有的话需要重新登录
            while (ceshi.status_code != 200) and (ceshi.status_code != 203):#'!=200'意味着重定向到了登录页面，登录页面的验证码请求是加密的其他url，无法从此登录
                print(str(ceshi.status_code))
                print('登录失败，正在重新登录')
                time.sleep(10)
                proxycontent = requests.get('http://api.xdaili.cn/xdaili-api//privateProxy/applyStaticProxy?spiderId=0a4b8956ad274e579822b533d27f79e1&returnType=1&count=1')#接入混拨代理
                print('已获取IP')
                proxylist = re.findall('(.*?)\\r\\n',proxycontent.text)
                print('正在检查IP')
                proxylist = checkip(proxylist)
                for l in range(0,len(proxylist)):
                    proxylist[l] = {"http":"http://"+ proxylist[l],}
                print(proxylist)
                while (len(proxylist) <=4):
                    print('有效ip数目不足，需等待15秒重新提取')
                    time.sleep(10)
                    proxycontent = requests.get('http://api.xdaili.cn/xdaili-api//privateProxy/applyStaticProxy?spiderId=0a4b8956ad274e579822b533d27f79e1&returnType=1&count=1')
                    print('已获取IP')
                    proxylist = re.findall('(.*?)\\r\\n',proxycontent.text)
                    print('正在检查IP')
                    proxylist = checkip(proxylist)
                    for j in range(0,len(proxylist)):
                        proxylist[j] = {"http":"http://" + proxylist[j],}
                    print(proxylist)
                header = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'}#设置UA假装是浏览器
                header['User-Agent'] = random.choice(UAlist)
                r = requests.Session()#开启会话
                r.proxies = random.choice(proxylist)
                main()
                ceshi = r.get('http://www.okooo.com/soccer/match/?date=2017-01-01',headers = header,verify=False,allow_redirects=False,timeout = 31)
            print('登录成功')
            print('准备进入：' + i)
            if n == 0:
                dangtianbisai(i,int(beginpoint.startgame))#从断点比赛开始爬取数据，并在屏幕打印出用时
            else:
                dangtianbisai(i)
            n = 1
            r.close()#关闭会话
            error = False
    except Exception as e:
        print('Error:',e)
        print('IP不可用，需要重新提取')
        time.sleep(15)
        n = 0
        error = True
    


