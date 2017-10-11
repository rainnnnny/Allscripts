import xlrd
import os
import random
import re
import time

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "hellodjango.settings")
import django
django.setup()

'''
Django 版本大于等于1.7的时候，需要加上下面两句
import django
django.setup()
否则会抛出错误 django.core.exceptions.AppRegistryNotReady: Models aren't loaded yet.
'''

from main.models import Express
import django.utils.timezone as timezone


oAll = Express.objects.all()


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

add(1, 2)


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

print(numFunc(3, 6))



def singleton(cls, *args):
    instances = {}
    def getinstance():
        if cls not in instances:
            instances[cls] = cls(*args)
        return instances[cls]
    return getinstance

@singleton
class Myclass():
