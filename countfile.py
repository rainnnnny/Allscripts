import os

mypath = "D:/phpserver/website"

def CountFile(perPath):
    num = 0
    for sName in os.listdir(perPath):
        sPath = os.path.join(perPath, sName)
        if os.path.isdir(sPath):
            CountFile(sPath)
        elif os.path.isfile(sPath):
            num += 1

    if num >= 500:
        print(perPath, num)

CountFile(mypath)
