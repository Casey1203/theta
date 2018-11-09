# coding: utf-8
# @Time    : 18-11-8 下午5:20
# @Author  : jia

import tensorflow as tf
from theta.util.file_loader import load_mnist_img_data, load_mnist_label_data, load_mnist_data
import numpy as np

def convert_to_one_hot(y, C):
	return np.eye(C)[y.reshape(-1)]


def neural_net(x, weight, bias):
	layer_1 = tf.add(tf.matmul(x, weight['h1']), bias['b1'])
	layer_2 = tf.add(tf.matmul(layer_1, weight['h2']), bias['b2'])
	out_layer = tf.add(tf.matmul(layer_2, weight['out']), bias['out'])
	return out_layer

def train(data_input, class_label, num_epoch, batch_size):
	# 定义placeholder
	m, n = data_input.shape
	label_num = 10
	num_input = 784
	num_hidden_layer_1 = 256
	num_hidden_layer_2 = 256
	X = tf.placeholder(dtype=tf.float32, shape=[None, num_input], name='data_input')
	y = tf.placeholder(dtype=tf.float32, shape=[None, label_num], name='class_label')

	# 定义variable
	weight = {
		'h1': tf.Variable(tf.random_normal([n, num_hidden_layer_1], name='h1')),
		'h2': tf.Variable(tf.random_normal([num_hidden_layer_1, num_hidden_layer_2], name='h2')),
		'out': tf.Variable(tf.random_normal([num_hidden_layer_2, label_num], name='out'))
	}
	bias = {
		'b1': tf.Variable(tf.random_normal([num_hidden_layer_1], name='b1')),
		'b2': tf.Variable(tf.random_normal([num_hidden_layer_2], name='b2')),
		'out': tf.Variable(tf.random_normal([label_num], name='out'))
	}
	logits = neural_net(X, weight, bias)

	tf.add_to_collection('network-output', logits)
	# 定义Loss function
	cross_entropy = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
	)
	# 定义optimizer op
	train_model = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cross_entropy)
	# 定义准确率op
	correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	# 开始训练

	init = tf.initialize_all_variables()
	saver = tf.train.Saver()
	model_path = '../../model/mnist.ckpt'
	with tf.Session() as sess:
		sess.run(init)
		batch_offset = 0
		for i in xrange(num_epoch):
			batch_x, batch_y = data_input[batch_offset: batch_offset + batch_size], class_label[batch_offset: batch_offset + batch_size]
			sess.run(train_model, feed_dict={X: batch_x, y: batch_y})
			if i % 100 == 0:
				loss, acc = sess.run([cross_entropy, accuracy], feed_dict={X: batch_x, y: batch_y})
				print 'step: %s, mini_batch loss: %s, training accuracy: %s' % (i, loss, acc)
				saver.save(sess, model_path, global_step=i + 1)
		batch_offset += batch_size







def evaluate(model_path, meta_path, x_test, y_test):
	saver = tf.train.import_meta_graph(meta_path)
	with tf.Session() as sess:
		saver.restore(sess, model_path)
		predict = tf.get_collection('network-output')[0]
		graph = tf.get_default_graph()
		X = graph.get_operation_by_name('data_input').outputs[0] # placeholder
		y = graph.get_operation_by_name('class_label').outputs[0] # placeholder
		logits = sess.run(predict, feed_dict={X: x_test, y: y_test})
		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(logits, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		print 'evaluate accuracy: ', sess.run(accuracy, feed_dict={X: x_test, y: y_test})



def main():
	# x_file_name = '../../data/train-images-idx3-ubyte.gz'
	# y_file_name = '../../data/train-labels-idx1-ubyte.gz'
	# data_input, x_head = load_mnist_img_data(x_file_name)
	# class_label, y_head = load_mnist_label_data(y_file_name)
	file_name = '../../data/mnist.npz'
	x_train, y_train, x_test, y_test = load_mnist_data(file_name)
	y_train = convert_to_one_hot(y_train, 10)
	# train(data_input=x_train, class_label=y_train, num_epoch=500, batch_size=128)
	y_test = convert_to_one_hot(y_test, 10)
	model_path = '../../model/mnist.ckpt-501'
	meta_path = '../../model/mnist.ckpt-501.meta'
	evaluate(model_path, meta_path, x_test, y_test)




if __name__ == '__main__':
	main()
