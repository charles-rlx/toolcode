from sklearn import preprocessing
import numpy as np
import csv
from sklearn.manifold import TSNE


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

def changeDataDimensional(X, dim):
    tsne = TSNE(n_components=dim, init='pca', random_state=0)
    result = tsne.fit_transform(X)
    return result

def nor(X):
    X_normalized = preprocessing.normalize(X, norm='l2')
    return X_normalized


if __name__ == '__main__':
    print("Here is dataprocess file")
