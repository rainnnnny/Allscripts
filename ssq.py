import random



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


res = [[]]
print(get())
