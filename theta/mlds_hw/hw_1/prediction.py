# coding: utf-8
# @Time    : 18-11-21 下午6:17
# @Author  : jia

from reader import load_testing_data
from rnn_2 import RNNModel
import pickle, os
import tensorflow as tf

def main():
	path = '../../data/testing_data.csv'
	test_sentences = load_testing_data(path)
	n_step = 40
	hidden_size = 400
	n_layer = 1
	batch_size = 5
	num_sampled = 2000
	grad_clip = 2.5
	data_path = '../../data/Holmes_Training_Data'
	vocab_path = os.path.join(data_path, 'vocab.pkl')
	vocab_dict = pickle.load(open(vocab_path, 'rb'))
	vocab, vocab_id_map = vocab_dict['vocab'], vocab_dict['vocab_id_map']
	vocab_size = len(vocab)
	model = RNNModel(n_step, hidden_size, n_layer, batch_size, vocab_size, num_sampled)
	model_path = '../../model'
	saver = tf.train.Saver()
	latest_checkpoint = saver.last_checkpoints()
	with tf.Session() as sess:
		saver.restore(sess, latest_checkpoint)

		prediction = []
