#Model.py
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.metrics import plot_precision_recall_curve
import numpy as np

#liner_logistic_regression
def model_liner_logistic_regression(X_train, Y_train):
    model_LLR = linear_model.LogisticRegression(penalty='l2',dual=False,C=10.0,n_jobs=1,random_state=20,fit_intercept=True, solver='newton-cg', max_iter=1000, tol=0.00001)
    model_LLR.fit(X_train,Y_train)
    fig = plot_precision_recall_curve(estimator=model_LLR, X=X_train, y=Y_train, sample_weight=None, response_method='auto')
    return model_LLR, fig

#SVC
def model_SVC(X_train, Y_train):
    model_SVC = SVC(gamma='auto', kernel='sigmoid', C=200, decision_function_shape='ovo', tol=1e-3)
    model_SVC.fit(X_train, Y_train)
    # fig = plot_precision_recall_curve(estimator=model_SVC, X=X_train, y=Y_train, sample_weight=None, response_method='auto')
    return model_SVC


