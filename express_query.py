import urllib.request
import time
import json

ID = 896089843631
URL = "http://www.kuaidi100.com/query?type=shunfeng&postid=%s&id=1&valicode=&temp=0.6068259679410473" % ID

def main():
    while 1:
        data = urllib.request.urlopen(URL).read()
        data = data.decode('UTF-8')
        data = json.loads(data)

        try:
            if data['data'][0]['time'] != '2017-12-14 20:07:00':
                print(time.asctime(), data['data'][0]['context'], data['data'][0]['time'])
            else:
                print('.', end='', flush=True)
        except IndexError:
            print(data['data'])

        time.sleep(1)
    

main()