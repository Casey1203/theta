# coding: utf-8
# @Time    : 18-10-29 下午5:12
# @Author  : jia

import numpy as np
from stump import build_stump, stump_classify
from theta.util.file_loader import load_simple_data, load_horse_colic_dataset
from theta.metrics.roc import plot_roc

def calculate_error_rate(G_m, data_input, class_label):
	predict_label = np.sign(G_m(data_input))
	return len((predict_label == data_input))

def calculate_alpha(error_rate):
	return 0.5 * np.log((1 - error_rate) / error_rate)


class Adaboost:
	def __init__(self, M):
		self.G = {}
		self.alpha = {}
		self.M = M

	def fit(self, data_input, class_label):
		N, n = np.shape(data_input) # N：样本数, n: 样本特征
		D = np.ones((N,)) / N # init weight vector
		aggregate_label = 0.
		for m in xrange(self.M):
			stump, error_rate = build_stump(data_input, class_label, D)
			stump['alpha'] = calculate_alpha(error_rate)
			self.G[m] = stump
			D_prime = np.exp(-stump['alpha'] * stump['predict_value'] * class_label) * D
			D = D_prime / D_prime.sum()
			aggregate_label += self.G[m]['alpha'] * self.G[m]['predict_value']
			aggregate_error_rate = (np.multiply(np.sign(aggregate_label) != class_label, np.ones((N,)))).sum() / N
			print 'iter_num: %s, total error rate: %s' % (m, aggregate_error_rate)
			if aggregate_error_rate == 0.:
				break
		return self.G, aggregate_label

	def evaluate(self, data_input, class_label):
		N = np.shape(data_input)[0]
		num_stump = len(self.G)
		aggregate_label = np.zeros((N, ))
		error_vector = np.ones((N, ))
		for m in xrange(num_stump):
			output_label = stump_classify(data_input, self.G[m]['dimension'], self.G[m]['threshold_value'], self.G[m]['threshold_inequality'])
			aggregate_label += self.G[m]['alpha'] * output_label
		error_vector[np.sign(aggregate_label) == class_label] = 0
		error_rate = float(error_vector.sum()) / N
		return error_rate

	def predict(self, data_input):
		N = np.shape(data_input)[0]
		num_stump = len(self.G)
		aggregate_label = np.zeros((N, ))
		for m in xrange(num_stump):
			output_label = stump_classify(data_input, self.G[m]['dimension'], self.G[m]['threshold_value'], self.G[m]['threshold_inequality'])
			aggregate_label += self.G[m]['alpha'] * output_label
		return np.sign(aggregate_label)


def test():
	# data_input, class_label = load_simple_data()
	data_input, class_label = load_horse_colic_dataset('../data/horseColicTraining2.txt')
	model = Adaboost(10)
	_, aggregate_label = model.fit(data_input, class_label)
	# print model.predict(np.array([[5, 5], [0, 0]]))
	# data_input, class_label = load_horse_colic_dataset('../data/horseColicTest2.txt')
	# test_error_rate = model.evaluate(data_input, class_label)
	# print 'test_error_rate: %s' % test_error_rate
	plot_roc(aggregate_label, class_label)

if __name__ == '__main__':
	test()