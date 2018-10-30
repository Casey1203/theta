# coding: utf-8
# @Time    : 18-10-30 下午3:33
# @Author  : jia
import numpy as np

def stump_classify(data_input, dim, threshold_value, threshold_inequality):
	# 指定data_input的某个dimension，用该dim下的数值去和threshold比较
	# 符号取决于threshold_inequality
	output_label = np.ones((np.shape(data_input)[0])) # init output label
	if threshold_inequality == 'lower_than':
		output_label[data_input[:, dim] <= threshold_value] = -1.
	else:
		output_label[data_input[:, dim] > threshold_value] = -1.
	return output_label


def build_stump(data_input, class_label, D):
	m, n = np.shape(data_input) # m： 样本个数，n：特征个数
	num_step = 10 # 切分点的个数
	min_error = np.inf
	best_stump = {}
	best_stump['predict_value'] = np.zeros((m, ))
	for i in xrange(n):
		feat_min = data_input[:, i].min()
		feat_max = data_input[:, i].max()
		step_size = float(feat_max - feat_min) / num_step
		for j in xrange(num_step):
			for threshold_inequality in ['lower_than', 'higher_than']:
				threshold_value = feat_min + j * step_size
				predict_value = stump_classify(data_input, i, threshold_value, threshold_inequality)
				error_array = np.ones((m, )) # init error rate
				error_array[predict_value == class_label] = 0.
				weighted_error = D.dot(error_array)
				if weighted_error < min_error:
					print 'threshold_value: %s, threshold_inequality: %s, dimension: %s, weighted_error: %s' % (
						threshold_value, threshold_inequality, i, weighted_error
					)
					best_stump['threshold_value'] = threshold_value
					best_stump['threshold_inequality'] = threshold_inequality
					best_stump['dimension'] = i
					best_stump['predict_value'] = predict_value
					min_error = weighted_error
	return best_stump, min_error