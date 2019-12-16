#starter.py
import dataprocess
import measurement
import models
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing



if __name__ == '__main__':
	#DATA
	X_train, Y_train = dataprocess.get_data(path = '../data/trainingset.csv')
	X_cross, Y_cross = dataprocess.get_data(path = '../data/crossset.csv')
	X_test, Y_test = dataprocess.get_data(path = '../data/testset.csv')
	X_train_cross, Y_train_cross = dataprocess.get_data(path='../data/training_cross.csv')
	#normalize:
	X_train = dataprocess.nor(X_train)
	X_cross = dataprocess.nor(X_cross)
	X_test = dataprocess.nor(X_test)
	X_train_cross = dataprocess.nor(X_train_cross)

	# X_train = dataprocess.addDimensional(X_train)
	# X_cross = dataprocess.addDimensional(X_cross)
	# X_test = dataprocess.addDimensional(X_test)

	#model selction
	# model_lrr, fig_pr_lrr = models.model_liner_logistic_regression(X_train, Y_train)
	# Y_cross_pred = model_lrr.predict(X_cross)
	model_svc = models.model_SVC(X_train, Y_train)
	# # print(model_svc.class_weight_)
	Y_train_pred = model_svc.predict(X_train)
	Y_cross_pred = model_svc.predict(X_cross)
	Y_test_pred = model_svc.predict(X_test)
	# #F1 score:
	print(measurement.f1(Y_train, Y_train_pred))
	print(measurement.f1(Y_cross, Y_cross_pred))
	print(measurement.f1(Y_test, Y_test_pred))

	#draw figue
	# plt.show(fig_pr_lrr)
	# fig_db_svc = measurement.draw_decision_boundary_2D(X_train, Y_train, X_train, Y_train)
	# plt.show(fig_db_svc)
	# fig_db_svc_2 = measurement.draw_decision_boundary_2D(X_train, Y_train, X_cross, Y_cross)
	# plt.show(fig_db_svc_2)
	# measurement.plot_valication_curve(X_train_cross, Y_train_cross)
	# measurement.plot_learning_curve(X_train_cross, Y_train_cross)
