# coding: utf-8
# @Time    : 18-11-26 下午3:09
# @Author  : jia
import tensorflow as tf
from rnn_2 import RNNModel
import time, os, pickle
import numpy as np
from reader import load_holmes_raw_data
from parser import clean_sentence, build_vocab, embedded_sentence_by_id
from gensim.models import KeyedVectors

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
	word_embedding = config['word_embedding']
	n_batch = len(data) / batch_size
	data = data[:n_batch * batch_size]
	data_x = data[:, :-1]
	data_y = data[:, 1:]
	# cpu_num = 6
	# sess_config = tf.ConfigProto(
	# 	device_count={'CPU': cpu_num},
	# 	inter_op_parallelism_threads=cpu_num,
	# 	intra_op_parallelism_threads=cpu_num,
	# 	log_device_placement=True
	# )
	with tf.Graph().as_default():
		with tf.Session() as sess:
			model = RNNModel(n_step, hidden_size, n_layer, batch_size, vocab_size, num_sampled, word_embedding)
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

def init_word_embedding(pretrained_word_vector, vocab, embedding_size):
	vocab_size = len(vocab)
	word_embedding = np.zeros((vocab_size, embedding_size), dtype=np.float32)
	for k, v in enumerate(vocab):
		if v in pretrained_word_vector:
			word_embedding[k] = pretrained_word_vector[v]
		else:
			word_embedding[k] = np.random.randn(embedding_size)
	return word_embedding

def main():
	stop_word = open('../../data/stopwords.txt', 'r').read().split('\n')
	# word_vector = KeyedVectors.load_word2vec_format('../../model/word_vector.bin', binary=True)
	n_step = 38
	hidden_size = 300
	n_layer = 1
	batch_size = 32
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
	num_sampled = int(0.5 * len(vocab))
	print 'num_sampled:', num_sampled
	word_vector = KeyedVectors.load_word2vec_format('../../model/word_vector.bin', binary=True)
	word_embedding = init_word_embedding(word_vector, vocab, hidden_size)
	config = {
		'n_step': n_step, 'hidden_size': hidden_size, 'n_layer': n_layer, 'batch_size': batch_size,
		'vocab_size': len(vocab), 'num_sampled': num_sampled, 'learning_rate': learning_rate, 'grad_clip': grad_clip,
		'n_epoch': n_epoch, 'keep_prob': 0.9, 'save_every': 100, 'save_dir': 'model/',
		'word_embedding': word_embedding
	}

	train(id_s_list, config)

if __name__ == '__main__':
	main()