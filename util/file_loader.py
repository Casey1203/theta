# coding: utf-8
# @Time    : 18-10-24 下午4:48
# @Author  : jia

import numpy as np
import gzip, struct
import matplotlib.pyplot as plt

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


def load_mnist_img_data(x_file_name):
	bin_file = open(x_file_name, 'rb') # 读取二进制文件
	buffers = bin_file.read()
	head = struct.unpack_from('>IIII', buffers, 0) # 读取前4个整数，返回一个元祖
	offset = struct.calcsize('>IIII') # 定位到data开始的位置

	img_num = head[1]

	fig = plt.figure()
	magic, numImages, imgRows, imgCols = struct.unpack_from(">IIII", buffers, 0)
	data_input = []
	for i in xrange(img_num):
		print i
		tmp = struct.unpack_from('>784B', buffers, offset)
		im = np.array(tmp).reshape(28, 28)
		fig.add_subplot(111)
		plt.imshow(im, cmap='gray')
		plt.show()
		data_input.append(im)
		offset += struct.calcsize('>784B')
	data_input = np.array(data_input)
	bin_file.close()

	return data_input, head

def load_mnist_label_data(y_file_name):
	bin_file = open(y_file_name, 'rb') # 读取二进制文件
	buffers = bin_file.read()
	head = struct.unpack_from('>II', buffers, 0) # 读取前2个整数
	label_num = head[1]
	offset = struct.calcsize('>II') # 定位到label开始的位置
	num_string = '>' + str(label_num) + 'B' # fmt格式：'>60000B'
	labels = struct.unpack_from(num_string, buffers, offset) # 取label数据
	bin_file.close()
	labels = np.reshape(labels, [label_num]) # 转型为列表(一维数组)
	return labels, head


def load_mnist_data(file_name):
	f = np.load(file_name)
	x_train, y_train = f['x_train'], f['y_train']
	x_test, y_test = f['x_test'], f['y_test']
	x_num, x_width, x_height = x_train.shape
	x_train = x_train.reshape([x_num, x_width * x_height])
	x_num, x_width, x_height = x_test.shape
	x_test = x_test.reshape([x_num, x_width * x_height])
	f.close()
	return x_train, y_train, x_test, y_test


