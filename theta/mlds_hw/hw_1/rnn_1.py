# coding: utf-8
# @Time    : 18-11-15 上午10:59
# @Author  : jia

# 思路：
# 以空格为分界线，前面一部分句子跑一个lstm，取空格前最后一个state作为句子的vector
# 后面一部分句子跑一个lstm，取空格后面第一个state作为句子的vector
# 把两个vector拼接起来，接一层fc，加一个softmax，得到具体输出的词

import tensorflow as tf
from gensim.models import KeyedVectors
from reader import load_holmes_raw_data
import os
from parser import sentence_parser, embedded_sentence, embedded_word

def build_encoder(X, hidden_size, n_layer, n_step, n_input, keep_prob, batch_size):
	# X_forward = tf.placeholder(shape=[batch_size, None, hidden_size]) # batch_size, time_steps, hidden_size
	# X_backward = tf.placeholder(shape=[batch_size, None, hidden_size]) # batch_size, time_steps, hidden_size
	# # y = tf.placeholder(dtype='float', shape=[None, n_input]) # batch_size, n_input

	cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=0., state_is_tuple=True)
	if keep_prob:
		cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
	# stacked rnn
	cell = tf.nn.rnn_cell.MultiRNNCell([cell] * n_layer, state_is_tuple=True)

	init_state = cell.zero_state(batch_size, tf.float32)

	X = tf.nn.dropout(X, keep_prob)

	X = tf.reshape(X, [-1, n_input]) # reshape to (n_step * batch_size, n_input)
	X = tf.split(X, n_step, 0) # split to get a list of n_step tensors of shape(batch_size, n_input)

	outputs, states = tf.nn.dynamic_rnn(cell, X, initial_state=init_state, dtype=tf.float32, time_major=True)

	return outputs, states

class RNNModel(object):
	def __init__(self, batch_size, hidden_size, n_layer, n_step, keep_prob, n_input, lr):
		y = tf.placeholder(shape=[batch_size, n_input])
		X_forward = tf.placeholder(shape=[batch_size, n_step, hidden_size]) # batch_size, time_steps, hidden_size
		X_backward = tf.placeholder(shape=[batch_size, n_step, hidden_size]) # batch_size, time_steps, hidden_size

		outputs_forward, states_forward = build_encoder(X_forward, hidden_size, n_layer, n_step, n_input, keep_prob, batch_size)
		outputs_backward, states_backward = build_encoder(X_backward, hidden_size, n_layer, n_step, n_input, keep_prob, batch_size)
		fc_weights = tf.Variable(tf.random_normal([2 * hidden_size, n_input]), name='out')
		fc_bias = tf.Variable(tf.random_normal([n_input]), name='out')

		output = tf.concat([outputs_forward[-1, :, :], outputs_backward[-1, :, :]])
		logits = tf.matmul(output, fc_weights) + fc_bias

		loss = tf.losses.mean_squared_error(y, logits)

		train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)

		self._loss = loss
		self._train_op = train_op

		self._input_data_forward = X_forward
		self._input_data_backward = X_backward
		self._target = y
		# self._final_state =

	@property
	def input_data_forward(self):
		return self._input_data_forward

	@property
	def input_data_backward(self):
		return self._input_data_backward

	@property
	def target(self):
		return self._target

	@property
	def loss(self):
		return self._loss

	@property
	def train_op(self):
		return self._train_op





def main():
	batch_size = 256
	hidden_size = 800
	n_layer = 2
	n_step = 28
	n_input = 300
	learning_rate = 0.001
	n_epoch = 2
	word_vector_size = 300
	keep_prob = 0.9
	stop_word = open('../../data/stopwords.txt', 'r').read().split('\n')
	word_vector = KeyedVectors.load_word2vec_format('../../model/word_vector.bin', binary=True)
	data_path = '../../data/Holmes_Training_Data'
	file_list = os.listdir(data_path)
	sentences_list = []
	for file_name in file_list:
		sentence_list, _ = load_holmes_raw_data(file_name)
		sentences_list += sentence_list
	n_sentences = len(sentences_list)
	n_batch = n_sentences / batch_size
	sentences_list = sentences_list[:n_batch * batch_size]
	model = RNNModel(batch_size, hidden_size, n_layer, n_step, keep_prob, n_input, learning_rate)
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	for epoch in xrange(n_epoch):
		for i in xrange(1, n_batch):
			batch_X_forward, batch_X_backward, batch_y = [], [], []
			sentence_batch = sentences_list[(i-1)*batch_size: i*batch_size]
			for sentence in sentence_batch:
				is_succeed, forward_sentence, backward_sentence, mid_word = sentence_parser(sentence, 10, stop_word)
				if is_succeed:
					embedded_forward_sentence = embedded_sentence(forward_sentence, stop_word, word_vector, word_vector_size)
					batch_X_forward.append(embedded_forward_sentence)
					embedded_backward_sentence = embedded_sentence(backward_sentence, stop_word, word_vector, word_vector_size)
					batch_X_backward.append(embedded_backward_sentence)
					embedded_mid_word = embedded_word(mid_word, stop_word, word_vector, word_vector_size)
					batch_y.append(embedded_mid_word)
			fetches = [model.loss, ]
			sess.run(train_op, feed_dict={X_forward: embedded_forward_sentence, })


