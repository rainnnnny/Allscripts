import logging


def getlogger(name):
	logger = logging.getLogger(name)

	if logger.hasHandlers():
		return logger

	formatter = logging.Formatter('%(asctime)s %(filename)s %(funcName)s %(levelname)s: %(message)s')
	#"%(asctime)s %(name)s [%(levelname)s] %(thread)d %(module)s %(funcName)s %(lineno)s: %(message)s"
	handler = logging.FileHandler('/home/null/var/log/%s.log'%name)
	handler.formatter = formatter
	
	logger.addHandler(handler)
	logger.setLevel(logging.INFO)

	return logger