# coding: utf-8

import tensorflow as tf
from scipy import spatial
import numpy as np

LEARNING_RATE = 0.000018
BATCH_SIZE = 256
DISPLAY_STEP = 100

WORD_VECTOR_SIZE = 300
import pickle


def predict_quality(list1, list2, threshold):
	pred_quality = 0.
	for i in xrange(BATCH_SIZE):
		cos = spatial.distance.cosine(list1[i], list2[i])
		if cos > threshold:
			pred_quality += 1
	return pred_quality / BATCH_SIZE



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

	def build_graph(self, x, config):

		keep_prob = train_config['keep_prob']

		rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden, state_is_tuple=True)
		if keep_prob < 1:
			rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=keep_prob)
		outputs, _ = tf.nn.dynamic_rnn(rnn_cell, x, dtype=tf.float32)
		return tf.matmul(outputs[:, -1, :], self.weights['out']) + self.bias['out']

	def train(self, train_X, train_Y, train_config):
		batch_size = train_config['batch_size']
		n_layer = train_config['n_layer']
		n_step = train_config['n_step']
		n_input = train_config['n_input']
		x = tf.placeholder(dtype='float', shape=[None, n_step, n_input]) # batch_size, sequence_length, frame_size
		y = tf.placeholder(dtype='float', shape=[None, n_input]) # batch_size, frame_size
		lr = train_config['lr']
		n_epoch = train_config['n_epoch']
		data_size = len(train_X)
		n_batch = data_size / batch_size + 1
		# X = tf.placeholder(dtype='float', shape=[None, n_step, n_input])
		# y = tf.placeholder(dtype='float', shape=[None, self.n_classes])
		# X_ = tf.transpose(X, [1, 0, 2]) # permuting batch_size and n_step
		# X_ = tf.reshape(X_, [-1, n_input]) # reshape to (n_step * batch_size, n_input)
		# X_ = tf.split(X_, n_step, 0) # split to get a list of n_step tensors of shape(batch_size, n_input)
		predict = self.build_graph(x, train_config)

		seperate_cost = tf.losses.cosine_distance(y, predict, axis=1)

		cost = tf.reduce_mean(seperate_cost)

		optimizer = tf.train.GradientDescentOptimizer(lr).minimize(seperate_cost)

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			for epoch in xrange(n_epoch):
				# lr = init_lr / (epoch + 1)
				for batch_number in xrange(n_batch):
					batch_index = batch_size * batch_number
					batch_X = train_X[batch_index: batch_index + batch_size, :, :]
					batch_Y = train_Y[batch_index: batch_index + batch_size]
					# batch_X = batch_X.reshape((batch_size, n_step, n_input))
					# batch_Y = batch_Y.reshape((batch_size, n_input))

					sess.run(optimizer, feed_dict={x: batch_X, y: batch_Y})

					if batch_number % DISPLAY_STEP == 0:
						p = sess.run(predict, feed_dict={x: batch_X})
						accuracy = predict_quality(p, batch_Y, 0.4)
						loss = sess.run(cost, feed_dict={x: batch_X, y: batch_Y})
						print 'Epoch: %s, batch number: %s, batch loss: %s, training accuracy: %s' % (epoch, batch_number, loss, accuracy)
				save_path = '../../model/holmes_lm-%s.ckpt' % epoch
				saver = tf.train.Saver()
				saver.save(sess, save_path)





if __name__ == '__main__':
	n_hidden = 800
	train_config = {
		'n_step': 2,
		'n_input': 300,
		'keep_prob': 0.9,
		'n_layer': 2,
		'batch_size': 256,
		'n_epoch': 2,
		'lr': 0.000018
	}

	model_config = {
		'n_hidden': 800,
		'n_classes': 300
	}
	training_embedding_data = pickle.load(open('../../data/embedding_train_data_-1.pkl', "rb"))
	train_X = np.array(training_embedding_data['embedding_train_X'])
	train_Y = np.array(training_embedding_data['embedding_train_Y'])

	rnn_model = RNN(model_config)
	rnn_model.train(train_X, train_Y, train_config)







