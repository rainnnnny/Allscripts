#-*- coding:utf-8 -*-
'''
Description :  
递归遍历目录下所有文件名(排除目录)，并逐行写入到指定文件中。
可以分别用py2或py3来执行，结果相同。
可以不带参数，或者 python xxxx <path> <writepath>
'''

'''
一些说明：
Python2: str -> (decode) -> unicode -> (encode) -> str
Python3: bytes -> (decode) -> str(unicode) -> (encode) -> bytes

官方文档表示，3.0后： 'All text is Unicode; however encoded Unicode is represented as binary data.
The type used to hold text is str, the type used to hold data is bytes'
'''

import sys
import os

try:
    PATH = sys.argv[1]
except IndexError:
    PATH = r'./'  # raw string, 表示不进行转义, 如果复制一个带反斜杠后面带数字或字母的路径, 不加上这个r就报错了

try:
    WRITE_PATH = sys.argv[2]
except IndexError:
    WRITE_PATH = 'abc'   # 指定要写入的文件名


PY2 = sys.version.startswith('2')


if PY2:
    # 不理解编码的人经常用这个当做万能药，这个确实也有用，但严重不推荐使用
    # import sys
    # reload(sys)  
    # sys.setdefaultencoding('utf8')
    # PATH = PATH.decode()  # 即使用了万能药这句也是要的


    # 记住原则，在python内处理文本字符串，永远保证是unicode类型，关于'ignore'参数见第4篇
    PATH = PATH.decode('utf8', 'ignore')


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
    # Python2, 由于py2中概念的模糊, 可以直接用'w+'打开去写, 不过不编成utf8的话也是会抛UnicodeDecodeError的。
    # 检查 "%s\n" % each 的类型可以看到是unicode，说明py2内部处理过程中也一直是unicode（废话）
    with open(WRITE_PATH, 'w+') as f:
        for each in res:
            f.write(("%s\n" % each).encode('utf8'))

else:
    # Python3, 可以用w+打开然后不编码直接写str(unicode)，不过那样结果很明显：非英文各种乱码。
    # 而编了码就转为了bytes类型，所以Python3想正确实现就必须用二进制方式打开 （wb+）
    # 如果打开方式和写入类型不对应，直接就抛TypeError了
    with open(WRITE_PATH, 'wb+') as f:
        for each in res:
            f.write(("%s\n" % each).encode('utf8'))