import urllib.request
import time
import json
import os
from pprint import pformat, pprint
import gevent.subprocess

import requests
from bs4 import BeautifulSoup as bs

import log

log = log.getlogger('watcher')

MYPLACE = ['guangzhou', 'shenzhen']


def getlabal(s):
    tmp = list(s)
    tmp.insert(1, '/')
    return "".join(tmp)


def check_who():
    wholist = os.popen("who | awk '{print $5}'").readlines()
    result = []
    if not wholist:
        return []
    for who in wholist:
        # who = '(14.16.225.186)\n'
        try:
            ip = who.strip()[1:-1]
        except ValueError:
            return ['only tty found: {}'.format(wholist)]
        
        result.append(crawl_ip_info(ip))

    return result


def crawl_ip_info(ip):
 
    url = 'http://api.db-ip.com/v2/free/%s' % ip

    data = requests.get(url).json()

    return data

def main():
    result = check_who()
    pprint(result)
    invader, myself, unknown = [], [], []

    for data in result:        
        try:
            city = data.get('city').lower()
        except:
            log.warn('Get city info failed! (%s)' % data)
            unknown.append(data)
            continue
        
        myplace = False
        for each in MYPLACE:
             if each == city:
                 myplace = True
                 break
        if myplace:
            myself.append([each, data.get('ipAddress')]) # only need show city if is myself
        else:
            invader.append(data)

    if invader:
        log.warn('Invader found: %s' % pformat(invader))
    if myself:
        log.info('Youself found: %s' % pformat(myself))
    if unknown:
        log.info('unknown: %s' % pformat(unknown))

if __name__ == '__main__':
    main()
