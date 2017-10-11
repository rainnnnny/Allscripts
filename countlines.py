import os

path = "./bstpl"

for sName in os.listdir(path):
	if sName == "assets":
		continue
	with open(path+"/"+sName) as file:
		try:
			print(sName, len(file.readlines()))
		except:
			print(sName, "error")