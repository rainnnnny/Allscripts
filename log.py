import logging
import os


def getlogger(name):
    logger = logging.getLogger(name)

    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter('%(asctime)s %(filename)s %(funcName)s %(levelname)s: %(message)s')
    #"%(asctime)s %(name)s [%(levelname)s] %(thread)d %(module)s %(funcName)s %(lineno)s: %(message)s"
        path = '~/var/log/%s.log'%name
        if not  os.path.isfile(path):
            f = open(path, 'w')
            f.close()

        handler = logging.FileHandler(path)
    handler.formatter = formatter
    
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    return logger
