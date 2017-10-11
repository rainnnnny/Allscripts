import os


PATH = "C:/Users/rainn/AppData/Local/Packages/Microsoft.Windows.ContentDeliveryManager_cw5n1h2txyewy/LocalState/Assets"

EXTNAME = ".jpg"

os.chdir(PATH)

for idx, sFileName in enumerate(os.listdir("./")):
	# if "." in sFileName:
	# 	sFileName = sFileName[:-4]
	
	if not sFileName.endswith(EXTNAME):
		os.rename(sFileName, "%s%s"%(idx, EXTNAME))
		print ("creating %s%s"%(idx, EXTNAME))
	else:
		print ("no file to change")

input()