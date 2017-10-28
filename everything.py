import xlrd
import os
import random
import re
import time
import math



#装饰器
def deco(func):
    def wrapper(a, b):
        st = time.time()
        func(a, b)
        et = time.time()
        print("run time:%s"%(st-et))
    return wrapper

@deco
def add(a, b):
    print(a+b)
# add(1, 2)



#闭包
s = "string in global"
num = 99

def numFunc(a, b):
    num = 100
    s = "string in addFunc"
    print( "print s in numFunc: ", s)
    print( "print num in numFunc: ", num)

    def addFunc(a, b):
        print( "print s in addFunc: ", s)
        print( "print num in addFunc: ", num)
        return "%d + %d = %d" %(a, b, a + b)

    return addFunc(a, b)

# print(numFunc(3, 6))



#单例
def singleton(cls, *args):
    instances = {}
    def getinstance():
        if cls not in instances:
            instances[cls] = cls(*args)
        return instances[cls]
    return getinstance



#range里用len是否影响效率
def tmp1():
    a = list(range(20000000))
    m=0
    t1 = time.time()
    for i in range(len(a)):
        m=1
    print(time.time()-t1)

    t2 = time.time()
    for i in range(20000000):
        m=1
    print(time.time()-t2)



#n人轮流报数，数到k出局，最后剩的一个
def tmp2():
    n, k = 100, 7
    ori = list(range(1, n+1))
    counter = 0
    out = []
    curMax = ori[-1]
    while len(ori) > 1:
        oneloop = 0
        while oneloop < k:
            counter += 1
            if counter > curMax:
                counter -= curMax
            if counter not in out:
                oneloop += 1

        out.append(counter)
        ori.remove(counter)
        curMax = ori[-1]
    return ori, out



#快排
def QuickSort(lList, start, end):
    if start >= end:
        return
    flag = lList[end]
    i, j = start, end - 1
    while i <= j:
        if lList[i] > flag:
            lList[j], lList[i] = lList[i], lList[j]
            j -= 1
        else:
            i += 1
    lList[j+1], lList[end] = lList[end], lList[j+1]
    print(lList, flag)
    QuickSort(lList, start, j)
    QuickSort(lList, j+2, end)

# a = list(range(70))
# random.shuffle(a)
# print(a)
# QuickSort(a, 0, len(a)-1)
# print(a)