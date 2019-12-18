import pandas as pd
import numpy as np
import pickle
import csv

with open('./models/english/ml_model_irrelevant-en.pickle', 'rb') as model_file:
    clf_ori = pickle.load(model_file)

with open('./data/original_model_features.csv', encoding = "utf-8") as feature_file:
    data = csv.reader(feature_file)
    next(data)
    vectors = []
    Y_true = []
    for row in data:
        vector = []
        # vectors.append(row[1:])
        for i in range(len(row)):
            if i == 0:
                pass
            elif i == 344:
                Y_true.append(int(row[i]))
            else:
                vector.append(int(row[i]))
        vectors.append(vector)

# vectors = np.array(vectors)
Y_true = np.array(Y_true)
Y_pred = clf_ori.predict(vectors)
# print(vectors)

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
    Accurancy = (TP + TN) / len(arr_true)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F = Precision * Recall * 2 / (Precision + Recall)
    return {"Accurancy":Accurancy, "Precision":Precision, "Recall":Recall, "F1":F, "TP":TP, "FP":FP, "FN":FN, "TN":TN}

print(f1(Y_true, Y_pred))


print(clf_ori.predict_proba(vectors))




