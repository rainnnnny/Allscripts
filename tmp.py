'''
crawler
'''

import urllib.request
import time

URL = "https://www.zhihu.com/question/68546899"

def xx1():
    while 1:
        data = urllib.request.urlopen(URL).read()
        data = data.decode('UTF-8')

        posa = data.find('关注者')
        numa = data[posa+59:posa+65]

        posb = data.find('被浏览')
        numb = data[posb+59:posb+67]

        print(time.asctime(), "  关注者: ", numa, "  被浏览: ", numb)

        time.sleep(0.5)

if __name__ == "__main__":
    main()
    