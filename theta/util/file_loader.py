# coding: utf-8
# @Time    : 18-10-24 下午4:48
# @Author  : jia

import numpy as np

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
