# coding: utf-8
# @Time    : 18-11-8 下午5:20
# @Author  : jia

import tensorflow as tf
from theta.util.file_loader import load_mnist_data


def build_model(data_input, class_label):
	m, n = data_input.shape
	X = tf.placeholder(shape=[None, m, n, None], name='data_input')
	y = tf.placeholder(shape=[None, m, None, None], name='class_label')
	


def train():
	pass


def evaluate():
	pass



def main():
	x_file_name = '../../data/train-images-idx3-ubyte.gz'
	y_file_name = '../../data/train-labels-idx1-ubyte.gz'
	data_input, class_label = load_mnist_data(x_file_name, y_file_name)
	print data_input
	print class_label


if __name__ == '__main__':
	main()
