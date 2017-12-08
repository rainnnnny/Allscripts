#-*- coding:utf-8 -*-

''' 
Python2: str -> decode -> unicode -> encode -> str
Python3: bytes -> decode -> str(unicode) -> encode -> bytes
unicode是py内部编码
py3去掉了unicode()函数
'''

PY2 = False

import os

if PY2:
    import sys  
    reload(sys)  
    sys.setdefaultencoding('utf8')   

fpath = 'abc'
PATH = r'./'    # raw, 表示不进行转义, 加上这个防止反斜杠或者中文之类导致报错

if PY2:
    # Py2如果不设为unicode类型，listdir接收到str报错，因为Python内部使用unicode，listdir需要的参数类型也必然为unicode
    PATH = unicode(PATH)


def getf(path):
    l = []
    res = os.listdir(path)
    for each in res:
        subpath = os.path.join(path, each)
        if os.path.isdir(subpath):
            l.extend(getf(subpath))
        else:
            l.append(each)

    return l

res = getf(PATH)

if PY2:
    # Python2, str本身即拥有编码(被上面代码设置为默认utf-8)，所以可以直接写
    with open(fpath, 'w+') as f:
        for each in res:
            f.write("%s\n" % each)
else:
    # Python3, 由于直接写str就是unicode了，没有编码，而编了码就转为了bytes类型，所以Python3想实现就必须用二进制方式打开
    with open(fpath, 'wb+') as f:
        for each in res:
            f.write(("%s\n" % each).encode('utf8'))

        # f.write(str(res).encode('utf-8'))