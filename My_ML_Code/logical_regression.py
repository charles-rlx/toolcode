from sklearn import linear_model
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.model_selection import learning_curve
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt
import csv
from start import addDimensional
import numpy as np

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


	#training part:
X_train, Y_train = get_data(path = '../data/trainingset.csv')
X_cross, Y_cross = get_data(path = '../data/crossset.csv')
X_test, Y_test = get_data(path = '../data/testset.csv')
X_train = addDimensional(X_train)
X_cross = addDimensional(X_cross)
model = linear_model.LogisticRegression(penalty='l2',dual=False,C=10.0,n_jobs=1,random_state=20,fit_intercept=True, solver='newton-cg', max_iter=1000, tol=0.00001)
model.fit(X_train,Y_train)
# print(model)
print(np.shape(X_train))
#f1 score in crossset:
Y_pred = model.predict(X_cross)
print("crossset f1 score:"+str(f1_score(Y_cross, Y_pred)))
# print(precision_recall_fscore_support(Y_cross, Y_pred))
print(f1(Y_cross, Y_pred))
#f1 score in testset:
# Y_pred_test = model.predict(X_test)
# print("crossset f1 score:"+str(f1_score(Y_test, Y_pred_test)))


# from sklearn.svm import SVC
# clf = SVC(gamma='auto')
# clf.fit(X_train, Y_train)
# Y_pred_SVC = clf.predict(X_cross)
# print("crossset f1 score:"+str(f1_score(Y_cross, Y_pred_SVC)))

fig = plot_precision_recall_curve(estimator=model, X=X_train, y=Y_train, sample_weight=None, response_method='auto')
plt.show(fig)

