import sklearn.linear_model as sk_linear
from sklearn.metrics import f1_score
import csv


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


	#training part:
X_train, Y_train = get_data(path = '../data/trainingset.csv')
X_test, Y_test = get_data(path = '../data/crossset.csv')
model = sk_linear.LogisticRegression(penalty='l2',dual=False,C=1.0,n_jobs=1,random_state=20,fit_intercept=True)
model.fit(X_train,Y_train)
acc=model.score(X_test,Y_test)
print('evaluation:',acc)