import urllib.request
import time
import json
import os

from pprint import pformat
from bs4 import BeautifulSoup as bs

import mymail
import log

log = log.getlogger('watcher')

MYPLACE = ['guangzhou', 'shenzhen']


def getlabal(s):
    tmp = list(s)
    tmp.insert(1, '/')
    return "".join(tmp)


def check_who():
    wholist = os.popen('who').readlines()
    if not wholist:
        return [], []
    for who in wholist:
        # null     pts/2        2018-01-21 03:43 (14.16.225.186)
        try:
            ip = who[who.index('(')+1:who.index(')')]
        except ValueError:
            return [], 'only tty found: {}'.format(wholist)
        url = 'https://db-ip.com/%s' % ip

        data = urllib.request.urlopen(url).read()
        s=bs(data, 'lxml')

        th = s('th')
        td = s('td')

        result = {}
        for i, each in enumerate(th):
            item = each.get_text().strip()
            if item in ['ASN', 'City', 'Country']:
                result[item] = td[i].get_text().strip()
        result['ip'] = ip

        try:
            city = result.get('City').lower()
        except:
            log.warn('Get city info failed!')
            os._exit(0)

        invader, myself = [], []
        myplace = False
        for each in MYPLACE:
            if each in city:
                myplace = True
                break
        if myplace:
            myself.append(result)
        else:
            invader.append(result)

        return invader, myself

def main():
    invader, myself = check_who()
    if invader:
        log.warn('Invader found:\n %s \n will send mail' % pformat(invader))
        mymail.send(password=mymail.password, msg=pformat(invader), subject="Warning, Invader Found.")
    if myself:
        log.info('Youself found: %s ' % str(myself))
    if not invader and not myself:
        log.info('Fine, a lonely planet.')

if __name__ == '__main__':
    main()
