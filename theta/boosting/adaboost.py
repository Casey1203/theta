# coding: utf-8
# @Time    : 18-10-29 下午5:12
# @Author  : jia

import numpy as np

def build_stump(data_input, class_label, D):
	m, n = np.shape(data_input)
	num_step = 10.
	min_error = np.inf


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
		N, n = np.shape(data_input) # N：样本数

		D_1 = np.ones((N,)) / N # 权值初始化
		# calculate error rate
		self.G[1] = build_stump(data_input, class_label, D_1)

		e_1 = calculate_error_rate(self.G[1], data_input, class_label)
		self.alpha[1] = calculate_alpha(e_1)
		Z_1 = (D_1 * np.exp(-self.alpha[1] * class_label * self.G[1](data_input))).sum()

		D_m = D_1 * np.exp(-self.alpha[1] * class_label * self.G[1](data_input)) / Z_1

		for m in xrange(2, self.M):
			self.G[m] = build_stump(data_input, class_label, D_m)
			e_m = calculate_error_rate(self.G[m], data_input, class_label)
			self.alpha[m] = calculate_alpha(e_m)
			Z_m = (D_m * np.exp(-self.alpha[m] * class_label * self.G[m](data_input))).sum()
			D_m *= np.exp(-self.alpha[m] * class_label * self.G[m](data_input)) / Z_m

	def evaluate(self, data_input, class_label):
		model_output = np.zeros(class_label.shape)
		for m in xrange(self.M):
			model_output += self.alpha[m] * self.G[m](data_input)
		model_output = np.sign(model_output)
		error_rate = (model_output == class_label) / len(class_label)
		return error_rate

	def predict(self, data_input):
		N = np.shape(data_input)
		model_output = np.zeros((N,))
		for m in xrange(self.M):
			model_output += self.alpha[m] * self.G[m](data_input)
		model_output = np.sign(model_output)
		return model_output








