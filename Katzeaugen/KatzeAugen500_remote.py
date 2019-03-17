#实时监控澳客网赔率，经过测试，500彩票网的赔率更新与betbrains是保持一致的，而澳客网的赔率更新时间数据和500彩票网一样，所以基本上可以认为没有问题
#由于澳客网是动态加载，所以本代码从每场比赛的欧赔加载中找到原始地址,通过请求ajax的原始地址获取数据
#原始地址每个请求返回30个公司赔率数据，这样每场比赛大约6到13个请求，每周大约500到600场比赛，则最多不到8000个请求。
#如果ajax请求的服务器承受能力跟单个公司历史赔率页面相同，那么每秒50个请求来算，同步一周的比赛大约需要3分钟左右的时间
#经试验，请求主页的ajax不需要登陆，但是请求下一周的比赛还是要登录的，所以顺序应该如常，进入主页，登陆，进入日期，获取链接，然后接下来做————20190112
#ajax下来的网页解码方式是unicode-escape，与其他网页不同————20190112
#由于完整爬完某一天所有比赛的历史赔率要很长时间，所以此监控程序只监控各场比赛当前赔率，所以要想获得某一场比赛的完整赔率需要提前一个月开始监控————20190316
#先给未来一个月的比赛的网址备案，有的比赛带有赔率的把公司的名称也备案，得到未来一个月的比赛的一个数据库，也就是大约2000张表。
# 然后更新时再把这一个月的表爬一遍，如果数据有更新通过merge来更新，并把更新的数据传回到本机，本机再将新的数据和老数据叠加构成变盘数据库。————20190316
#本程序打算用多进程的方式管理ip池，一旦某个ip失效，则通过另一个进程在ip池中剔除此ip，这样防止了重复利用无效ip。如果ip数量不足，则所有请求暂时挂起，待新ip提取完毕再继续执行————20190316
#需要一个本地的mysql数据库
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


r = requests.Session()
header = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'}
header['Referer'] = 'http://www.okooo.com/soccer/'#必须加上这个才能进入足球日历
header['Upgrade-Insecure-Requests'] = '1'#这个也得加上
#经试验，请求主页的ajax不需要登陆，但是请求下一周的比赛还是要登录的，所以顺序应该如常，进入主页，登陆，进入日期，获取链接，然后接下来做
def dateRange(start, end, step=1, format="%Y-%m-%d"):#生成日期列表函数，用于给datelist赋值
    strptime, strftime = datetime.strptime, datetime.strftime
    days = (strptime(end, format) - strptime(start, format)).days + 1
    return [strftime(strptime(start, format) + timedelta(i), format) for i in range(0, days, step)]


def jinruriqi(date):
    global r
    global header
    global proxy
    wangye = r.get('http://www.okooo.com/soccer/match/?date='+date,headers = header,verify=False,allow_redirects=False,timeout = 9.5,proxies=proxy)
    content1 = wangye.content.decode('gb18030')#取出wangye的源代码
    sucker1 = '/soccer/match/.*?/odds/'
    bisaiurl = re.findall(sucker1,content1)#获得当天的比赛列表
    return bisaiurl
def riqiliebiao():
    today = time.strftime("%Y-%m-%d")#今天
    nextmonth = datetime.strftime(datetime.now()+timedelta(35),"%Y-%m-%d")#下个月，威廉一些重要比赛甚至提前一个多月就出了
    datelist = dateRange(today,nextmonth)#生成日期列表
    bisailist = list()
    for i in datelist:#获得一个月内所有比赛的url
        bisailist = bisailist + jinruriqi(i)
    return bisailist

def danchangbisai(url):#对单场比赛进行ajax请求以获得当前赔率并入库


def huoqupeilv(bisailist):#对所有比赛的网址进行爬取
    





content = a.content.decode('unicode-escape')#注意一旦请求下来后，ajax下来的网页解码方式是这个，与其他网页不同

