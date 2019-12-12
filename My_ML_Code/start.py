#coding='utf-8'
#make data visualise
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import preprocessing
from time import time
import csv
from mpl_toolkits.mplot3d import Axes3D


#GET X MATRIX AND Y MARTIX
def get_data(path):
    dataFile = open(path, 'r')
    lines = csv.reader(dataFile)
    next(lines)
    data_lists = list(lines)
    X, Y = [], []
    for i in range(len(data_lists)):
        temp_x = data_lists[i][1:5]
        temp_x = list(map(int, temp_x))
        temp_x.append(int(data_lists[i][6]))
        X.append(temp_x)

        temp_y = data_lists[i][7]
        Y.append(int(temp_y))

    return X, Y

#PLOT DATA
def plot_embedding_2D(data,label,title):
    x_min,x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max-x_min)
    fig  = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.plot(data[i, 0], data[i, 1],'o', color = plt.cm.Set1(label[i]))
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig

def plot_embedding_3D(data,label,title):
    x_min,x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max-x_min)
    fig  = plt.figure()
    ax1 = plt.axes(projection='3d')
    for i in range(data.shape[0]):
        ax1.scatter(data[i,0], data[i,1], data[i,2],color = plt.cm.Set1(label[i]))#3D
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig

def square(x):
    return x**2

def listMulti(list, x):
    for i in range(len(list)):
        list[i]=list[i]*x

    return list

def addDimensional(X):
    X_new=[]
    X_shape=np.shape(X)
    for x in X:
        temp_x = list(map(square,x))
        for l in range(0,X_shape[1]):
            if len(x[l+1:]) == 0:
                break
            else:
                temp_x = temp_x+listMulti(x[l+1:],x[l])
        x = x+temp_x
        X_new.append(x)
    return X_new
    

def main():
    X_train, Y_train = get_data(path = '../data/trainingset.csv')
    print(X_train[0:10])
    X_train = preprocessing.normalize(X_train, norm='l1')
    print(X_train[0:10])
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(X_train)
    fig = plot_embedding_2D(result, Y_train,
                         't-SNE embedding of the digits (time %.2fs)'
                         % (time() - t0))
    plt.show(fig)

# 
if __name__ == '__main__':
    a = addDimensional([[1,2,3],[4, 5, 6]])
    # print(a)


