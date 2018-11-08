# coding: utf-8
# @Time    : 18-11-8 下午5:20
# @Author  : jia

import tensorflow as tf
from theta.util.file_loader import load_mnist_img_data, load_mnist_label_data, load_mnist_data


def train(data_input, class_label, num_epoch, batch_size):
	# 定义placeholder
	m, n = data_input.shape
	label_num = 10
	X = tf.placeholder(dtype=tf.float32, shape=[None, n], name='data_input')
	y = tf.placeholder(dtype=tf.float32, shape=[None, label_num], name='class_label')

	# 定义variable
	W = tf.Variable(tf.zeros([n, label_num]))
	b = tf.Variable(tf.zeros([label_num]) + 0.1)

	predict_y = tf.nn.softmax(tf.matmul(X, W) + b)
	# 定义Loss function
	cross_entropy = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=predict_y)
	)
	# 定义optimizer
	train_model = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cross_entropy)
	# 开始训练

	init = tf.initialize_all_variables()
	with tf.Session() as sess:
		sess.run(init)
		batch_offset = 0
		for i in xrange(num_epoch):
			batch_x, batch_y = data_input[batch_offset: batch_offset + batch_size], class_label[batch_offset: batch_offset + batch_size]
			sess.run(train_model, feed_dict={'X': batch_x, 'y': batch_y})
			if i % 100 == 0:
				correct_prediction = tf.equal(tf.argmax(predict_y, 1), tf.argmax(y, 1))
				accuracy = tf.reduce_mean(correct_prediction)
				print sess.run(accuracy, feed_dict=)
		batch_offset += batch_size


def evaluate(X, y):
	pass



def main():
	# x_file_name = '../../data/train-images-idx3-ubyte.gz'
	# y_file_name = '../../data/train-labels-idx1-ubyte.gz'
	# data_input, x_head = load_mnist_img_data(x_file_name)
	# class_label, y_head = load_mnist_label_data(y_file_name)
	file_name = '../../data/mnist.npz'
	x_train, y_train, x_test, y_test = load_mnist_data(file_name)



if __name__ == '__main__':
	main()