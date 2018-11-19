# coding: utf-8
# @Time    : 18-11-19 下午6:42
# @Author  : jia

import tensorflow as tf

class RNNModel(object):
	def __init__(self, n_step, hidden_size, n_layer, batch_size, vocab_size, num_sampled):
		self.x = tf.placeholder(tf.int32, shape=[None, n_step-1])
		self.y = tf.placeholder(tf.int32, shape=[None, n_step-1])

		self.keep_prob = tf.placeholder(tf.float32)

		self.cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
		self.cell = tf.nn.rnn_cell.DropoutWrapper(self.cell, output_keep_prob=self.keep_prob)

		self.cell = tf.nn.rnn_cell.MultiRNNCell([self.cell] * n_layer)
		self.init_state = self.cell.zero_state(batch_size, tf.float32)

		with tf.name_scope('embed'):
			embedding = tf.get_variable('embedding', shape=[vocab_size, hidden_size], dtype=tf.float32)
			inputs = tf.nn.embedding_lookup(embedding, self.x)
			inputs = tf.nn.dropout(inputs, self.keep_prob)

		outputs = []

		state = self.init_state
		with tf.variable_scope('LSTM'):
			for time_step in xrange(len(n_step)-1):
				if time_step > 0:
					tf.get_variable_scope().reuse_variables()
				(output, state) = self.cell(inputs[:, time_step, :], state)
				outputs.append(output)
		self.final_state = state

		output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, hidden_size])

		with tf.name_scope('nce_loss'):
			proj_w = tf.get_variable('proj_w', shape=[vocab_size, hidden_size])
			proj_b = tf.get_variable('proj_b', shape=[vocab_size])

			labels = tf.reshape(self.y, [-1, 1])
			# NCE loss
			loss = tf.nn.nce_loss(proj_w, proj_b, labels, output, num_sampled, vocab_size)

			self.cost = tf.reduce_sum(loss) / batch_size

		with tf.name_scope('softmax'):
			softmax_w = tf.transpose(proj_w)
			self.logits = tf.matmul(output, softmax_w) + proj_b

			self.probs = tf.nn.softmax(self.logits)

			seq_loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
				[self.logits],
				[tf.reshape(self.y, [-1])],
				[tf.ones([batch_size * (n_step - 1)], dtype=tf.float32)]
			)
			self.seq_loss = seq_loss
		with tf.name_scope('accuracy'):
			output_words = tf.argmax(self.probs, axis=1)
			output_words = tf.cast(output_words, tf.int32)
			self.equal = tf.equal(output_words, tf.reshape(self.y, [-1]))

def train(sentences_list, n_step, hidden_size, n_layer, batch_size, vocab_size, num_sampled, learning_rate, grad_clip, n_epoch):
	n_batch = len(sentences_list) / batch_size
	sentences_list = sentences_list[:n_batch * batch_size]
	with tf.Graph().as_default():
		with tf.Session() as sess:
			model = RNNModel(n_step, hidden_size, n_layer, batch_size, vocab_size, num_sampled)
			global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
			optimizer = tf.train.AdamOptimizer(learning_rate)
			tvars = tf.trainable_variables()
			grads, _ = tf.clip_by_global_norm(tf.gradients(model.cost, tvars), clip_norm=grad_clip)
			train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

			# grad_summaries = []
			saver = tf.train.Saver(tf.global_variables())
			# init all variable
			sess.run(tf.global_variables_initializer())

			for epoch in xrange(n_epoch):
				state = sess.run(model.init_state)
				for batch in xrange(n_batch):