# coding: utf-8
# @Time    : 18-11-1 上午11:19
# @Author  : jia

import matplotlib.pyplot as plt


def plot_roc(predict_strength, class_label):
	fig = plt.figure()
	fig.clf()
	ax = plt.subplot(111)

	num_pos_class = (class_label == 1).sum() # number of step to take in the y direction
	y_step = 1. / num_pos_class
	x_step = 1. / (len(class_label) - num_pos_class)
	sorted_indices = predict_strength.argsort() # from smallest to largest
	y_sum = 0 # calculate AUC
	cur = (1.0, 1.0) # cursor for plotting, start at 1, 1 and draw to 0, 0
	for index in sorted_indices:
		if class_label[index] == 1: # step down in y direction under label 1.0, which decrease the true positive rate
			del_x = 0
			del_y = y_step
		else: # step backward in x direction to decrease false positive rate
			del_x = x_step
			del_y = 0
			y_sum += cur[1] # add up a bunch of small rectangle

		cur = (cur[0] - del_x, cur[1] - del_y)
		ax.plot([cur[0], cur[0] - del_x], [cur[1], cur[1] - del_y], c='b')
	ax.plot([0, 1], [0, 1], 'b--')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC curve for Adaboost Horse Colic Detection System')
	ax.axis([0, 1, 0, 1]) # init figure size between 0 and 1
	plt.show()

	print 'the Area Under the Curve is: %s' % (y_sum * x_step)