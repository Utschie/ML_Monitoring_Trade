#本程序是datacleaning_2.0程序，通过把进程协同进一步，提高数据清洗效率
#1.5的程序洗出来数据预计要有4TB，所以将来全数据训练的时候要用此版本存储数据，所以2.0等之后再开发————20200624
#三个进程共同操作一个列表，这样就不会有某个进程过长其他进程一直等待的情况————20200629