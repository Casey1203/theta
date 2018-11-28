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
import os, time, pickle
from parser import clean_sentence, embedded_sentence_by_id, build_vocab, embedded_word_by_id
import numpy as np

def convert_to_one_hot(y, C):
	return np.eye(C, dtype=np.int32)[y.reshape(-1)]

def build_encoder(X, hidden_size, n_layer, keep_prob, batch_size, n_step, name):
	with tf.variable_scope(name) as scope:

		cell = tf.nn.rnn_cell.LSTMCell(hidden_size, forget_bias=1, state_is_tuple=True)
		cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
		# stacked rnn
		cell = tf.nn.rnn_cell.MultiRNNCell([cell] * n_layer, state_is_tuple=True)
		state = cell.zero_state(batch_size, dtype=tf.float32)

		outputs = []
		states = []
		# for time_step in xrange(n_step-1):
		# 	if time_step > 0:
		# 		tf.get_variable_scope().reuse_variables()
		# 	(output, state) = cell(X[:, time_step], state)
		# 	outputs.append(output)
		# 	states.append(state)
		outputs, states = tf.nn.dynamic_rnn(cell, X, initial_state=state, dtype=tf.float32)

	return outputs, states

class RNNModel(object):
	def __init__(self, n_step, hidden_size, n_layer, batch_size, vocab_size, num_sampled):
		self.keep_prob = tf.placeholder(tf.float32)
		self.y = tf.placeholder(dtype=tf.int32, shape=[None, vocab_size])
		self.X_forward = tf.placeholder(dtype=tf.int32, shape=[None, n_step]) # batch_size, time_steps
		self.X_backward = tf.placeholder(dtype=tf.int32, shape=[None, n_step]) # batch_size, time_steps
		embedding = tf.get_variable('embedding', shape=[vocab_size, hidden_size], dtype=tf.float32)
		# embedding = tf.Variable(np.identity(vocab_size, dtype=np.int32), dtype=tf.float32)
		with tf.name_scope('embed'):
			input_forward = tf.nn.embedding_lookup(embedding, self.X_forward)
			input_forward = tf.nn.dropout(input_forward, self.keep_prob)
			input_backward = tf.nn.embedding_lookup(embedding, self.X_backward)
			input_backward = tf.nn.dropout(input_backward, self.keep_prob)
		self.input_forward = input_forward
		outputs_forward, states_forward = build_encoder(input_forward, hidden_size, n_layer, self.keep_prob, batch_size, n_step, 'lstm1')
		outputs_backward, states_backward = build_encoder(input_backward, hidden_size, n_layer, self.keep_prob, batch_size, n_step, 'lstm2')
		output = tf.concat([outputs_forward[:, -1, :], outputs_backward[:, -1, :]], axis=1)
		# output = tf.concat([tf.reduce_sum(outputs_forward, axis=1), tf.reduce_sum(outputs_backward, axis=1)], axis=1)
		# output = tf.concat([outputs_forward[-1], outputs_backward[-1]], axis=1)

		# with tf.name_scope('nce_loss'):
		# 	# labels = tf.reshape(self.y, [-1, 1])
		# 	# NCE loss
		# 	loss = tf.nn.nce_loss(proj_w, proj_b, self.y, output, num_sampled, vocab_size)
		# 	self.nce_cost = tf.reduce_sum(loss) / batch_size
		fc_w = tf.Variable(tf.truncated_normal([2 * hidden_size, vocab_size], -0.1, 0.1), dtype=tf.float32, name='fc_w')
		fc_b = tf.Variable(tf.random_normal([vocab_size]), dtype=tf.float32, name='fc_b')
		self.logits = tf.matmul(output, fc_w) + fc_b
		self.probs = tf.nn.softmax(self.logits)
		with tf.name_scope('loss'):
			cost = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y)
			self.loss = tf.reduce_mean(cost)
		with tf.name_scope('accuracy'):
			# output_words = tf.argmax(self.probs, axis=1, output_type=tf.int32)
			# output_words = tf.cast(output_words, tf.int32)
			correct_prediction = tf.equal(tf.argmax(self.probs, 1), tf.argmax(self.y, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		self.fc_w = fc_w
		self.fc_b = fc_b
		self.output = output
		self.outputs_forward = outputs_forward
		self.outputs_backward = outputs_backward
		print 'rnn network already.'


def train(forward_id_s_list, backward_id_s_list, space_word_id_list, config):
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
	display_every = config['display_every']
	save_dir = config['save_dir']
	n_batch = len(forward_id_s_list) / batch_size
	forward_id_s_list = forward_id_s_list[:n_batch * batch_size]
	backward_id_s_list = backward_id_s_list[:n_batch * batch_size]
	space_word_id_list = space_word_id_list[:n_batch * batch_size]
	space_word_one_hot_list = convert_to_one_hot(space_word_id_list, vocab_size)
	cpu_num = 6
	config = tf.ConfigProto(
		device_count={'CPU': cpu_num},
		inter_op_parallelism_threads=cpu_num,
		intra_op_parallelism_threads=cpu_num,
		log_device_placement=True
	)
	with tf.Graph().as_default():
		with tf.Session(config=config) as sess:
			model = RNNModel(n_step, hidden_size, n_layer, batch_size, vocab_size, num_sampled)
			global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
			optimizer = tf.train.AdamOptimizer(learning_rate)
			tvars = tf.trainable_variables()
			grads, _ = tf.clip_by_global_norm(tf.gradients(model.loss, tvars), clip_norm=grad_clip)
			train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
			# train_op = optimizer.minimize(model.loss, global_step=global_step)

			# grad_summaries = []
			saver = tf.train.Saver(tf.global_variables())
			# init all variable
			sess.run(tf.global_variables_initializer())

			for epoch in xrange(n_epoch):
				# state = sess.run(model.init_state)
				for i in xrange(0, n_batch):
					start = time.time()
					data_forward_batch = forward_id_s_list[(i) * batch_size: (i+1) * batch_size]
					data_backward_batch = backward_id_s_list[(i) * batch_size: (i+1) * batch_size]
					data_y_batch = space_word_one_hot_list[(i) * batch_size: (i+1) * batch_size]
					feed_dict = {model.X_forward: data_forward_batch, model.X_backward: data_backward_batch, model.y: data_y_batch, model.keep_prob: keep_prob}
					_, step, loss, accuracy, logits, y, fc_w, output, outputs_backward, fc_b, input_forward, prob = sess.run([train_op, global_step, model.loss, model.accuracy, model.logits, model.y, model.fc_w, model.output, model.outputs_backward, model.fc_b, model.input_forward, model.probs], feed_dict=feed_dict)
					print "training step {}, epoch {}, batch {}/{}, loss: {:.4f}, accuracy: {:.4f}, time/batch: {:.3f}".\
						format(step, epoch, i, n_batch, loss, accuracy, time.time()-start)
					# print 'logits', logits
					# print 'y', y
					# print 'fc_w'
					# print fc_w
					# print 'output'
					# print output
					# print 'outputs_backward'
					# print outputs_backward
					current_step = tf.train.global_step(sess, global_step)
					if current_step != 0 and current_step % display_every == 0:
						print np.argmax(logits, axis=1)
					if current_step != 0 and (current_step % save_every == 0 or (epoch == n_epoch - 1 and i == n_batch-1)):
						print np.argmax(logits, axis=1)
						checkpoint_path = os.path.join(save_dir, 'model.ckpt')
						path = saver.save(sess, checkpoint_path, global_step=global_step)
						print 'save model checkpoint to {}'.format(path)



def main():
	stop_word = open('../../data/stopwords.txt', 'r').read().split('\n')
	# word_vector = KeyedVectors.load_word2vec_format('../../model/word_vector.bin', binary=True)
	n_step = 20
	n_layer = 1
	batch_size = 200
	learning_rate = 0.1
	grad_clip = 0.25
	n_epoch = 200
	data_path = '../../data/Holmes_Training_Data'
	forward_embedding_data_path = os.path.join(data_path, 'forward_id_s_list.npy')
	backward_embedding_data_path = os.path.join(data_path, 'backward_id_s_list.npy')
	space_word_embedding_data_path = os.path.join(data_path, 'space_word_id_list.npy')
	vocab_path = os.path.join(data_path, 'vocab.pkl')
	if os.path.exists(forward_embedding_data_path) and os.path.exists(vocab_path):
		print 'load embedding data and vocab..'
		forward_id_s_list = np.load(forward_embedding_data_path)
		backward_id_s_list = np.load(backward_embedding_data_path)
		space_word_id_list = np.load(space_word_embedding_data_path)
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
		print 'remove stop word..'
		for i, sentence in enumerate(sentences_list):
			sentences_list[i] = clean_sentence(sentence, stop_word)
		# add start and end mark
		print 'add start and end mark...'
		for i, sentence in enumerate(sentences_list):
			sentences_list[i] = '<START> ' + sentence + ' <END>'
		# remove too long sentence or too short sentence
		print 'remove too long or too short sentence..'
		sentences_list_bk = []
		for i, sentence in enumerate(sentences_list):
			if 8 <= len(sentence.split()) <= n_step * 2:
				sentences_list_bk.append(sentence)
		sentences_list = sentences_list_bk
		vocab, vocab_id_map = build_vocab(sentences_list)
		print 'divide sentence into forward and backward sentence'
		# divide forward and backward sentence
		forward_sentences_list, backward_sentences_list = [], []
		space_word_list = []
		for i, sentence in enumerate(sentences_list):
			sentence_word = sentence.split()
			half_sentence_len = len(sentence_word) / 2
			forward_sentence = ' '.join(sentence_word[:half_sentence_len])
			space_word = sentence_word[half_sentence_len]
			backward_sentence = ' '.join(sentence_word[half_sentence_len+1:])
			forward_sentences_list.append(forward_sentence)
			backward_sentences_list.append(backward_sentence)
			space_word_list.append(space_word)
		print 'map sentence to id'
		# map sentence to id
		forward_id_s_list, backward_id_s_list, space_word_id_list = [], [], []
		for i, sentence in enumerate(forward_sentences_list):
			forward_id_s_list.append(embedded_sentence_by_id(sentence, vocab_id_map, n_step))
		for i, sentence in enumerate(backward_sentences_list):
			backward_id_s_list.append(embedded_sentence_by_id(sentence, vocab_id_map, n_step))
		for i, word in enumerate(space_word_list):
			space_word_id_list.append(embedded_word_by_id(word, vocab_id_map))
		forward_id_s_list = np.array(forward_id_s_list)
		backward_id_s_list = np.array(backward_id_s_list)
		space_word_id_list = np.array(space_word_id_list)
		print 'save embedding data and vocab..'
		np.save(os.path.join(data_path, 'forward_id_s_list.npy'), forward_id_s_list)
		np.save(os.path.join(data_path, 'backward_id_s_list.npy'), backward_id_s_list)
		np.save(os.path.join(data_path, 'space_word_id_list.npy'), space_word_id_list)

		pickle.dump({'vocab': vocab, 'vocab_id_map': vocab_id_map}, open(vocab_path, 'wb'))

	num_sampled = int(0.7 * len(vocab))
	hidden_size = 400
	print 'num_sampled:', num_sampled
	config = {
		'n_step': n_step, 'hidden_size': hidden_size, 'n_layer': n_layer, 'batch_size': batch_size,
		'vocab_size': len(vocab), 'num_sampled': num_sampled, 'learning_rate': learning_rate, 'grad_clip': grad_clip,
		'n_epoch': n_epoch, 'keep_prob': 1, 'display_every': 100, 'save_every': 500, 'save_dir': 'model/'
	}
	# forward_id_s_list = np.array([forward_id_s_list[0]])
	# backward_id_s_list = np.array([backward_id_s_list[0]])
	# space_word_id_list = np.array([space_word_id_list[0]])
	train(forward_id_s_list, backward_id_s_list, space_word_id_list, config)

if __name__ == '__main__':
	main()