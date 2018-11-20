# coding: utf-8
# @Time    : 18-11-19 下午6:42
# @Author  : jia

import tensorflow as tf
from parser import build_vocab, clean_sentence, embedded_sentence_by_id
from gensim.models import KeyedVectors
import os, time
from reader import load_holmes_raw_data
import numpy as np
import pickle

class RNNModel(object):
	def __init__(self, n_step, hidden_size, n_layer, batch_size, vocab_size, num_sampled):
		self.x = tf.placeholder(tf.int32, shape=[None, n_step-1])
		self.y = tf.placeholder(tf.int32, shape=[None, n_step-1])

		self.keep_prob = tf.placeholder(tf.float32)

		self.cell = tf.nn.rnn_cell.LSTMCell(hidden_size)
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
			for time_step in xrange(n_step-1):
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
			self.seq_loss = tf.reduce_sum(seq_loss) / batch_size
		with tf.name_scope('accuracy'):
			output_words = tf.argmax(self.probs, axis=1)
			output_words = tf.cast(output_words, tf.int32)
			self.equal = tf.equal(output_words, tf.reshape(self.y, [-1]))
		self.softmax_b = proj_b


def train(data, config):
	n_step = config['n_step']
	hidden_size = config['hidden_size']
	n_layer = config['n_layer']
	batch_size = config['batch_size']
	vocab_size = config['vocab_size']
	num_sampled = config['num_sampled']
	learning_rate = config['learning_rate']
	grad_clip = config['grad_clip']
	n_epoch = config['n_epoch']
	keep_prob = config['keep_prob']
	save_every = config['save_every']
	save_dir = config['save_dir']
	n_batch = len(data) / batch_size
	data = data[:n_batch * batch_size]
	data_x = data[:, :-1]
	data_y = data[:, 1:]
	with tf.Graph().as_default():
		with tf.Session() as sess:
			model = RNNModel(n_step, hidden_size, n_layer, batch_size, vocab_size, num_sampled)
			global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
			optimizer = tf.train.AdamOptimizer(learning_rate)
			tvars = tf.trainable_variables()
			grads, _ = tf.clip_by_global_norm(tf.gradients(model.seq_loss, tvars), clip_norm=grad_clip)
			train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

			# grad_summaries = []
			saver = tf.train.Saver(tf.global_variables())
			# init all variable
			sess.run(tf.global_variables_initializer())
			saver.restore(sess, 'model/model.ckpt-300')

			for epoch in xrange(n_epoch):
				state = sess.run(model.init_state)
				for i in xrange(1, n_batch):
					start = time.time()
					data_x_batch = data_x[(i-1) * batch_size: i * batch_size]
					data_y_batch = data_y[(i-1) * batch_size: i * batch_size]
					feed_dict = {model.x: data_x_batch, model.y: data_y_batch, model.keep_prob: keep_prob}
					_, step, loss, seq_loss, equal = sess.run([train_op, global_step, model.cost, model.seq_loss, model.equal], feed_dict=feed_dict)
					print "training step {}, epoch {}, batch {}/{}, loss: {:.4f}, seq_loss: {:.4f}, accuracy: {:.4f}, time/batch: {:.3f}".\
						format(step, epoch, i, n_batch, loss, seq_loss, np.mean(equal), time.time()-start)
					current_step = tf.train.global_step(sess, global_step)
					if current_step != 0 and (current_step % save_every == 0 or (epoch == n_epoch - 1 and i == n_batch-1)):
						checkpoint_path = os.path.join(save_dir, 'model.ckpt')
						path = saver.save(sess, checkpoint_path, global_step=current_step)
						print 'save model checkpoint to {}'.format(path)


def main():
	stop_word = open('../../data/stopwords.txt', 'r').read().split('\n')
	# word_vector = KeyedVectors.load_word2vec_format('../../model/word_vector.bin', binary=True)
	n_step = 40
	hidden_size = 400
	n_layer = 1
	batch_size = 50
	num_sampled = 2000
	learning_rate = 0.02
	grad_clip = 2.5
	n_epoch = 2
	data_path = '../../data/Holmes_Training_Data'
	embedding_data_path = os.path.join(data_path, 'id_s_list.npy')
	vocab_path = os.path.join(data_path, 'vocab.pkl')
	if os.path.exists(embedding_data_path) and os.path.exists(vocab_path):
		print 'load embedding data and vocab..'
		id_s_list = np.load(embedding_data_path)
		vocab_dict = pickle.load(open(vocab_path, 'rb'))
		vocab, vocab_id_map = vocab_dict['vocab'], vocab_dict['vocab_id_map']
	else:
		print 'extract embedding data and vocab..'
		file_list = os.listdir(data_path)
		sentences_list = []
		for file_name in file_list:
			sentence_list, _ = load_holmes_raw_data(os.path.join(data_path, file_name))
			sentences_list += sentence_list
		# remove stop word
		for i, sentence in enumerate(sentences_list):
			sentences_list[i] = clean_sentence(sentence, stop_word)
		# add start and end mark
		for i, sentence in enumerate(sentences_list):
			sentences_list[i] = '<START> ' + sentence + ' <END>'
		# remove too long sentence
		sentences_list_bk = []
		for i, sentence in enumerate(sentences_list):
			if 4 <= len(sentence.split()) <= n_step:
				sentences_list_bk.append(sentence)
		sentences_list = sentences_list_bk

		vocab, vocab_id_map = build_vocab(sentences_list)
		# map sentence to id
		id_s_list = []
		for i, sentence in enumerate(sentences_list):
			id_s_list.append(embedded_sentence_by_id(sentence, vocab_id_map, n_step))
		id_s_list = np.array(id_s_list)
		print 'save embedding data and vocab..'
		np.save(os.path.join(data_path, 'id_s_list.npy'), id_s_list)
		pickle.dump({'vocab': vocab, 'vocab_id_map': vocab_id_map}, open(vocab_path, 'wb'))
	config = {
		'n_step': n_step, 'hidden_size': hidden_size, 'n_layer': n_layer, 'batch_size': batch_size,
		'vocab_size': len(vocab), 'num_sampled': num_sampled, 'learning_rate': learning_rate, 'grad_clip': grad_clip,
		'n_epoch': n_epoch, 'keep_prob': 1, 'save_every': 50, 'save_dir': 'model/'
	}

	train(id_s_list, config)

if __name__ == '__main__':
	main()