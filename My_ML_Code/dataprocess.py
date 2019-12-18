from sklearn import preprocessing
import numpy as np
import csv
from sklearn.manifold import TSNE
import copy


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

def addDimensionalCube(X):
    X_new= []
    X_shape=np.shape(X)
    for i in range(0,X_shape[0]):
        aList = []
        x = X[i]
        temp_squared_x = list(map(square,x))
        for j in range(0,X_shape[1]):
            y = copy.deepcopy(X[i])
            aList = aList+listMulti(y,temp_squared_x[j])
        for a in range(0,X_shape[1]):
            if len(x[a+2:]) == 0:
                break
            else:
                for b in range(a+1,X_shape[1]):
                    if len(x[b+1:]) == 0:
                        break
                    else:
                        for c in range(b+1, X_shape[1]):
                            temp_ans = []
                            temp_ans.append(x[a]*x[b]*x[c])
                            aList = aList + temp_ans
        X_new.append(aList)
    return X_new

    # for i in range(0,X_shape[0]):
    #     temp_list_x = []
    #     for j in range(0,X_shape[1]):
    #         temp = X[i][j]**2
    #         temp_list_x = temp_list_x + listMulti(X[i],temp)
    #     X_new.append(temp_list_x)
    # return X_new
    #     temp=x**2
    #     temp_x = temp_x+[i*temp for i in X]
    #     print(temp_x)
    #     X_new.append(temp_x)
    # return X_shape

def changeDataDimensional(X, dim):
    tsne = TSNE(n_components=dim, init='pca', random_state=0)
    result = tsne.fit_transform(X)
    return result

def nor(X):
    X_normalized = preprocessing.normalize(X, norm='l2')
    return X_normalized


if __name__ == '__main__':
    print("Here is dataprocess file")
