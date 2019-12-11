from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import csv

def divideArr(arr=[]):
	X = []
	Y = []
	for row in arr:
		X.append(row[0])
		Y.append(row[1])
	X = np.array(X)
	Y = np.array(Y)
	return X, Y

def divideArr3(arr=[]):
	X = []
	Y = []
	Z = []
	for row in arr:
		X.append(row[0])
		Y.append(row[1])
		Z.append(row[2])
	X = np.array(X)
	Y = np.array(Y)
	Z = np.array(Z)
	return X, Y, Z

def setColor(arr=[]):
	newArr = []
	for i in range(len(arr)):
		if arr[i] == 0: 
			newArr.append('r')
		else:
			newArr.append('b')
	newArr = np.array(newArr)
	return newArr


def draw_two_dim(XY, R):
	pca = PCA(n_components=2)
	newXY = pca.fit_transform(XY)
	X_axis, Y_axis = divideArr(newXY)
	R = setColor(R)
	plt.scatter(X_axis,Y_axis, s=75, c=R, alpha=.5)
	plt.show()

def draw_three_dim(XYZ, R):
	pca = PCA(n_components=3)
	newXYZ = pca.fit_transform(XYZ)
	X_axis, Y_axis, Z_axis = divideArr3(newXYZ)
	R = setColor(R)
	fig = plt.figure()
	ax = Axes3D(fig)
	ax.scatter(X_axis, Y_axis, Z_axis, c=R)
	plt.show()



PATH_Trainingset_CSV = "../data/trainingset.csv"
CSV_FILE = open(PATH_Trainingset_CSV,'r',encoding="utf-8")
csv_reader = csv.reader(CSV_FILE)
next(csv_reader)
trainingset_arr = []
for row in csv_reader:
	trainingset_arr.append([[int(row[1]),int(row[2]),int(row[3]),int(row[4]),int(row[6])],int(row[7])])

Feature_Arr, R = divideArr(trainingset_arr)


# draw_two_dim(Feature_Arr,R) #2D
draw_three_dim(Feature_Arr,R) #3D

