import random
import log
import time


log = log.getlogger('luck')


def get():
    red = list(range(1, 34))
    blue = range(1, 17)
    res = [[]]
    for _ in range(6):
        x = random.choice(red)
        red.remove(x)
        res[0].append(x)
    res[0].sort()
    res.append(random.choice(blue))
    return res

t1 = time.time()
trytimes = 0
res = get()
while res != [[8,13,17,22,26,30], 10]:
    res = get()
    trytimes += 1
    if trytimes % 1000 == 0:
        print(trytimes)
log.info("that's it, %s, %s, %s" % (res, trytimes, time.time() - t1))

after = get()
print("that's it", trytimes)
log.info("after: " % after)
