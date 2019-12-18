#measurement
import dataprocess
import models

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
# import sklearn.gaussian_process.GaussianProcessClassifier as GP_C

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
    model = models.model_SVC(X, y)
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
    return {"Precision":Precision, "Recall":Recall, "F1":F}

def plot_valication_curve(X, y):
    #---------For SVC-------------
    params_range = np.arange(1,200)
    train_scores, test_scores = validation_curve(SVC(gamma='auto', kernel='sigmoid', decision_function_shape='ovo', tol=1e-3), X, y,"C", params_range, cv=4, scoring='f1')
    
    #--------For GaussianProcessClassifier --------------
    # params_range = np.arange(1,200)
    # train_scores, test_scores = validation_curve(GP_C(multi_class = 'one_vs_one'), X, y, "max_iter_predict", params_range, cv=4, scoring='f1')
    # print(train_scores)
    # print(test_scores)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.title("Validation Curve with SVM")
    plt.xlabel(r"$\gamma$")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.plot(params_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.plot(params_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.plot(params_range, train_scores_mean-test_scores_mean, label="cut",
                 color="red", lw=lw)
    # plt.semilogx(params_range, train_scores_mean, label="Training score",
    #              color="darkorange", lw=lw)
    # plt.fill_between(params_range, train_scores_mean - train_scores_std,
    #                  train_scores_mean + train_scores_std, alpha=0.2,
    #                  color="darkorange", lw=lw)
    # plt.semilogx(params_range, test_scores_mean, label="Cross-validation score",
    #              color="navy", lw=lw)
    # plt.fill_between(params_range, test_scores_mean - test_scores_std,
    #                  test_scores_mean + test_scores_std, alpha=0.2,
    #                  color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()

def plot_learning_curve(X,y):
    train_sizes = [0.1,0.33,0.66,1.0]
    train_sizes, train_scores, test_scores= learning_curve(SVC(gamma='auto', kernel='rbf', C = 30, decision_function_shape='ovo', tol=1e-3), X, y,train_sizes=train_sizes, cv=4, scoring='f1')

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    plt.legend(loc="best")
    plt.show()



def f1_present(arr_true, arr_pred):
    result = f1(arr_true, arr_pred)
    print("##################")
    print("precision: "+ str(result['Precision']))
    print("Recall: "+ str(result['Recall']))
    print("F1: "+ str(result['F1']))
    print("TP:"+str(result['TP'])+"  "+"FP:"+str(result['FP'])+"  "+"FN:"+str(result['FN'])+"  "+"TN:"+str(result['TN']))
    # print("##################")

def grid_search(X,y):
    param_grid = {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf'], decision_function_shape:['ovo'], tol:[1e-3]}
    grid=GridSearchCV(SVC(),param_grid=param_grid,cv=5, scoring = 'f1')
    grid.fit(X,y)
    grid.grid_scores_,grid.best_params_,grid.best_score_






















