#starter.py
import dataprocess
import measurement
import models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss,zero_one_loss
from sklearn import preprocessing



if __name__ == '__main__':
    #-------GET DATA FROM CSV FILE-------------
    X_train, Y_train = dataprocess.get_data(path = '../data/trainingset.csv')
    X_cross, Y_cross = dataprocess.get_data(path = '../data/crossset.csv')
    X_test, Y_test = dataprocess.get_data(path = '../data/testset.csv')
    X_train_cross, Y_train_cross = dataprocess.get_data(path='../data/train_and_cross.csv')
    #Check size of data
    # print(len(Y_train))
    # print(len(Y_cross))
    # print(len(Y_test))
    # print(len(Y_train_cross))

    #-------Preprocess DATA-------------
    # Add more features for data
    # X_train = dataprocess.addDimensional(X_train)
    # X_cross = dataprocess.addDimensional(X_cross)
    # X_test = dataprocess.addDimensional(X_test)
    # X_train = dataprocess.addDimensionalCube(X_train)
    # X_cross = dataprocess.addDimensionalCube(X_cross)
    # X_test = dataprocess.addDimensionalCube(X_test)
    #normalize:
    # X_train = dataprocess.nor(X_train)
    # X_cross = dataprocess.nor(X_cross)
    # X_test = dataprocess.nor(X_test)
    # X_train_cross = dataprocess.nor(X_train_cross)
    # print(X_train[1])

    #--------Selection Classify Models -----------
    #model selction
    # model_lrr, fig_pr_lrr = models.model_liner_logistic_regression(X_train, Y_train)
    # Y_cross_pred = model_lrr.predict(X_cross)
    # model_svc = models.model_SVC(X_train, Y_train)
    model_Bayes = models.model_Bayes(X_train, Y_train)
    # model_Forest = models.model_Random_Forest(X_train, Y_train)
    # # # print(model_svc.class_weight_)
    # Y_train_pred = model_svc.predict(X_train)
    # Y_cross_pred = model_svc.predict(X_cross)
    # Y_test_pred = model_svc.predict(X_test)
    Y_train_pred = model_Bayes.predict(X_train)
    Y_cross_pred = model_Bayes.predict(X_cross)
    Y_test_pred = model_Bayes.predict(X_test)
    # Y_train_pred = model_Forest.predict(X_train)
    # Y_cross_pred = model_Forest.predict(X_cross)
    # Y_test_pred = model_Forest.predict(X_test)
    #--------Measurement for Models(Number) ---------
    # # #F1 score:
    # a=model_svc.fit_status_
    # print(a)
    print(measurement.f1(Y_train, Y_train_pred))
    print(measurement.f1(Y_cross, Y_cross_pred))
    print(measurement.f1(Y_test, Y_test_pred))
    print("AUC score: "+str(roc_auc_score(Y_test, Y_test_pred)))
    print("Accuracy: "+ str(accuracy_score(Y_test, Y_test_pred)))
    print("LOSS: "+ str(brier_score_loss(Y_test,Y_test_pred)))
    print("LOSS 2: "+ str(zero_one_loss(Y_test,Y_test_pred)))
    # print(np.shape(X_test))
    # A = [[1,2,3],[2,2,2]]
    # c=dataprocess.addDimensionalCube(A)
    # print(c)
    # print(len(c[0]))
    # print(len(c[1]))
    #--------Measurement for Models(Diagram) -----------
    # plt.show(fig_pr_lrr)
    # fig_db_svc = measurement.draw_decision_boundary_2D(X_train, Y_train, X_train, Y_train)
    # plt.show(fig_db_svc)
    fig_db_svc_2 = measurement.draw_decision_boundary_2D(X_train_cross, Y_train_cross, X_test, Y_test)
    plt.show(fig_db_svc_2)
    # measurement.plot_valication_curve(X_train_cross, Y_train_cross)
    # measurement.plot_learning_curve(X_train_cross, Y_train_cross)







