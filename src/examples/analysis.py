import numpy as np

filename = "./parameters-v1.txt"

params = np.zeros(540)

with open(filename, "r") as filestream:
	i = 0
	list_ = []
	for line in filestream:
		if (i==0 or i > 2000):
			i = i+1
			continue
		elif (i%2==1):
			list_.extend([float(x) for x in line.split(",")])
			i = i+1
		elif (i%2==0):
			list_.extend([float(x) for x in line.split(",")])
			params = params+np.array(list_)
			list_ = []
			i = i+1
mean = params/1000

params = np.zeros(540)
variance = np.zeros(540)

with open(filename, "r") as filestream:
	i = 0
	list_ = []
	for line in filestream:
		if (i==0 or i > 2000):
			i = i+1
			continue
		elif (i%2==1):
			list_.extend([float(x) for x in line.split(",")])
			i = i+1
		elif (i%2==0):
			list_.extend([float(x) for x in line.split(",")])
			params = params+(np.array(list_)-mean)**2
			list_ = []
			i = i+1
print((params/999).mean())
