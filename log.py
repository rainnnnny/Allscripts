import logging


def getlogger(name):
	logger = logging.getLogger(name)

	if logger.hasHandlers():
		return logger

	formatter = logging.Formatter('%(asctime)s %(filename)s %(levelname)s: %(message)s')
	handler = logging.FileHandler('/home/null/var/log/%s.log'%name)
	handler.formatter = formatter
	
	logger.addHandler(handler)
	logger.setLevel(logging.INFO)

	return logger