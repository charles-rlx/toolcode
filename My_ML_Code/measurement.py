#measurement
import dataprocess
import models

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# draw 2D figure
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

# draw 3D figure
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

# draw 2D decision boundary figure
def draw_decision_boundary_2D(X, y, X_cross, y_cross):
    X = dataprocess.changeDataDimensional(X, 2)
    X_cross = dataprocess.changeDataDimensional(X_cross, 2)
    model, fig_pr = models.model_liner_logistic_regression(X, y)
    fig  = plt.figure()
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    step = 0.02
    x_min_max, y_min_max = np.meshgrid(np.arange(x_min, x_max, step),np.arange(y_min, y_max, step))
    
    Z = model.predict(np.c_[x_min_max.ravel(), y_min_max.ravel()])
    Z = Z.reshape(x_min_max.shape)
    cs = plt.contourf(x_min_max, y_min_max, Z, cmap=plt.cm.Paired)
    plt.axis("tight")

    for i in range(X_cross.shape[0]):
        plt.scatter(X_cross[i, 0], X_cross[i, 1], color=plt.cm.Set1(y_cross[i]), cmap=plt.cm.Paired, s=20, edgecolor='k')
    plt.title('Decision Boundary')
    return fig

# compute f1 score
def f1(arr_true, arr_pred):
    TP, FP, FN, TN = 0, 0, 0, 0
    for i in range(len(arr_true)):      
        if arr_pred[i] == 1:
            if arr_true[i] == 1:
                TP+=1
            else:
                FP+=1 
        else:
            if arr_true[i] == 1:
                FN+=1
            else:
                TN+=1
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F = Precision * Recall * 2 / (Precision + Recall)
    return Precision, Recall, F

