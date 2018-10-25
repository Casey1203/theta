# coding: utf-8
import numpy as np
from sklearn import svm
from smo import outer_loop, calc_w
import matplotlib.pyplot as plt


def sk_svm(data_input, class_label, C=10000):
	clf = svm.SVC(kernel='linear', C=C)
	clf.fit(data_input, class_label)
	print clf.coef_
	print clf.intercept_
	return clf.coef_[0], clf.intercept_

def handcraft_svm(data_input, class_label, C=10000):
	b, alphas = outer_loop(data_input, class_label, C=C, toler=0.0001, maxIter=5000)
	w = calc_w(alphas, class_label, data_input)

	print 'b: %s' % b
	print 'w:', w
	print 'alpha:', alphas
	return w, b


if __name__ == '__main__':
	data_input = np.array([[1, 2], [2, 3], [3, 3], [2, 1], [3, 2]])
	class_label = np.array([1, 1, 1, -1, -1]).reshape((5, 1))

	# w, b = sk_svm(data_input, class_label, 10000)
	w, b = handcraft_svm(data_input, class_label, 10000)
	plt.scatter([i[0] for i in data_input], [i[1] for i in data_input], c=class_label.reshape((len(class_label,))))

	xasix = np.linspace(0, 3.5)
	a = -w[0] / w[1]
	y_sep = a * xasix - b / w[1]


	plt.plot(xasix, y_sep, 'k-')

	plt.show()