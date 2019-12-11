#coding='utf-8'
#make data visualise
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from time import time
import csv
from mpl_toolkits.mplot3d import Axes3D


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


def plot_embedding(data,label,title):
    x_min,x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max-x_min)
    fig  = plt.figure()
    # ax1 = plt.axes(projection='3d')#3D
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        # ax1.scatter(data[i,0], data[i,1], data[i,2],color = plt.cm.Set1(label[i]))#3D
        plt.plot(data[i, 0], data[i, 1],'o', color = plt.cm.Set1(label[i]))#2D
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig

def main():
    X_train, Y_train = get_data(path = 'data/trainingset.csv')
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(X_train)
    fig = plot_embedding(result, Y_train,
                         't-SNE embedding of the digits (time %.2fs)'
                         % (time() - t0))
    plt.show(fig)


if __name__ == '__main__':
    # train_X, train_Y = get_data(path = 'data/trainingset.csv')
    # print(train_X[1:5])
    # print(train_Y[1:5])
    main()
