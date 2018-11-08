# coding: utf-8
# @Time    : 18-10-24 下午4:48
# @Author  : jia

import numpy as np
import gzip

def load_dataset(file_name):
	data_mat = []
	label_mat = []
	fr = open(file_name)
	for line in fr.readlines():
		line_arr = line.strip().split('\t')
		data_mat.append([float(line_arr[0]), float(line_arr[1])])
		label_mat.append(float(line_arr[2]))
	data_arr = np.array(data_mat)
	label_arr = np.array(label_mat)
	label_arr = label_arr.reshape((len(label_arr), 1))
	return data_arr, label_arr


def load_simple_data():
	data_input = np.array([[1., 2.1], [2., 1.1], [1.3, 1.], [1., 1.], [2., 1.]])
	class_label = np.array([1., 1., -1., -1., 1.])

	return data_input, class_label


def load_horse_colic_dataset(file_name):
	# num_feat = len(open(file_name).readline().split('\t')) # split by tab
	fr = open(file_name)
	data_mat = []
	label_mat = []
	for line in fr.readlines():
		line_arr = line.strip().split('\t')
		data_mat.append([float(x) for x in line_arr[:-1]])
		label_mat.append(int(float(line_arr[-1])))
	data_arr = np.array(data_mat)
	label_arr = np.array(label_mat)
	return data_arr, label_arr


def load_mnist_data(x_file_name, y_file_name):
	data_mat = []
	label_mat = []
	with gzip.open(x_file_name, 'rb') as fr:
		for line in fr.readlines():
			print line
			data_mat.append(line)

	with gzip.open(y_file_name, 'rb') as fr:
		for line in fr.readlines():
			print line
			label_mat.append(line)
	data_arr = np.array(data_mat)
	label_arr = np.array(label_mat)
	return data_arr, label_arr
