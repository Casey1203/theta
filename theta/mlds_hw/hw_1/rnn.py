# coding: utf-8

import tensorflow as tf


LEARNING_RATE = 0.000018
num_epoch = 2
BATCH_SIZE = 256
DISPLAY_STEP = 100

WORD_VECTOR_SIZE = 300



class RNN:
	def __init__(self, config):
		self.n_hidden = config['n_hidden']
		self.n_classes = config['n_classes']

		self.weights = {
			'out': tf.Variable(tf.random_normal([self.n_hidden, self.n_classes]), name='out')
		}
		self.bias = {
			'out': tf.Variable(tf.random_normal([self.n_classes]), name='out')
		}

	def train(self, data_input, train_config):
		n_step = train_config['n_step']
		n_input = train_config['n_input']
		keep_prob = train_config['keep_prob']
		X = tf.placeholder(dtype='float', shape=[None, n_step, n_input], name='data_input')
		y = tf.placeholder(dtype='float', shape=[None, self.n_classes], name='class_label')

		rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden, state_is_tuple=True)
		if keep_prob < 1:
			rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=keep_prob)





