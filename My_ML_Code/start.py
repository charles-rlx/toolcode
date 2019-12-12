#coding='utf-8'
#make data visualise
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import preprocessing
from sklearn import linear_model

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

def get_model(X_train,Y_train):
    model = linear_model.LogisticRegression(penalty='l2',dual=False,C=1.0,n_jobs=1,random_state=20,fit_intercept=True, solver='newton-cg', max_iter=1000, tol=0.00001)
    model.fit(X_train,Y_train)

    return model

def draw_decision_boundary(X, y,):
    model = get_model(X, y)
    fig  = plt.figure()
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    step = 0.02
    x_min_max, y_min_max = np.meshgrid(np.arange(x_min, x_max, step),np.arange(y_min, y_max, step))
    
    Z = model.predict(np.c_[x_min_max.ravel(), y_min_max.ravel()])
    Z = Z.reshape(x_min_max.shape)
    cs = plt.contourf(x_min_max, y_min_max, Z, cmap=plt.cm.Paired)
    plt.axis("tight")

    for i in range(X.shape[0]):
        plt.scatter(X[i, 0], X[i, 1], color=plt.cm.Set1(y[i]), cmap=plt.cm.Paired, s=20, edgecolor='k')
    plt.title('Decision Boundary')
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
    X_test, Y_test = get_data(path = '../data/crossset.csv')
    X_train = preprocessing.normalize(X_train, norm='l1')
    # model = get_model(X_train, Y_train)
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(X_train)
    # fig = plot_embedding_2D(result, Y_train,'t-SNE embedding of the digits (time %.2fs)' % (time() - t0))
    # plt.show(fig)
    plt.show(draw_decision_boundary(result, Y_train))



if __name__ == '__main__':
    # a = addDimensional([[1,2,3],[4, 5, 6]])
    main()
    # print(a)


