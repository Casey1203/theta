# coding: utf-8

import tensorflow as tf


LEARNING_RATE = 0.000018
BATCH_SIZE = 256
DISPLAY_STEP = 100

WORD_VECTOR_SIZE = 300
import pickle



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


	def train(self, train_X, train_Y, train_config):
		n_step = train_config['n_step']
		n_input = train_config['n_input']
		batch_size = train_config['batch_size']
		n_layer = train_config['n_layer']
		keep_prob = train_config['keep_prob']
		lr = train_config['lr']
		n_epoch = train_config['n_epoch']
		data_size = len(train_X)
		n_batch = data_size / batch_size + 1
		init_lr = lr
		X = tf.placeholder(dtype='float', shape=[None, n_step, n_input], name='data_input')
		y = tf.placeholder(dtype='float', shape=[None, self.n_classes], name='class_label')

		rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden, state_is_tuple=True)
		if keep_prob < 1:
			rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=keep_prob)
		rnn_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell] * n_layer, state_is_tuple=True)
		state = rnn_cell.zero_state(batch_size, dtype='float')

		outputs = []
		with tf.VariableScope('RNN'):
			for time_step in xrange(n_step):
				if time_step > 0:
					tf.get_variable_scope().reuse_variables()
				(cell_output, state) = rnn_cell(data_input[:, time_step, :], state)
				outputs.append(cell_output)
			outputs, _ = tf.nn.static_rnn(rnn_cell, X, dtype=tf.float32)

		predict = tf.matmul(outputs[-1], self.weights['out']) + self.bias['out']
		seperate_cost = tf.losses.cosine_distance(y, predict)

		cost = tf.reduce_mean(seperate_cost)

		optimizer = tf.train.GradientDescentOptimizer(lr).minimize(seperate_cost)

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			for epoch in xrange(n_epoch):
				lr = init_lr / (epoch)
				for batch_number in xrange(n_batch):
					batch_index = batch_size * batch_number
					batch_X = train_X[batch_index: batch_index + batch_number]
					batch_Y = train_Y[batch_index: batch_index + batch_number]
					batch_X = batch_X.reshape((batch_size, n_step, n_input))
					batch_Y = batch_Y.reshape((batch_size, n_input))

					sess.run(optimizer, feed_dict={X: batch_X, y: batch_Y})

					if batch_number % DISPLAY_STEP == 0:
						correct_predict = []
						p = sess.run(predict, feed_dict={X: batch_X, y: batch_Y})




if __name__ == '__main__':
	n_hidden = 800
	train_config = {
		'n_step': 28,
		'n_input': 300,
		'keep_prob': 0.9,
		'n_layer': 2,
		'batch_size': 256,
		'n_epoch': 2
	}

	model_config = {
		'n_hidden': 800,
		'n_classes': 300
	}
	train_X, train_Y, = pickle.load(open('../../data/embedded_train_data_-1', "rb"))

	rnn_model = RNN(model_config)
	rnn_model.train(train_X, train_Y, train_config)







