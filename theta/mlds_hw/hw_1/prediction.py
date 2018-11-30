# coding: utf-8
# @Time    : 18-11-21 下午6:17
# @Author  : jia

from reader import load_testing_data_with_multiple_option, load_test_answer
from rnn_2 import RNNModel
import pickle, os
import tensorflow as tf
import numpy as np
from parser import clean_sentence, embedded_sentence_by_id, test_sentence_info

def main():
	n_step = 40
	path = '../../data/testing_data.csv'
	stop_word = open('../../data/stopwords.txt', 'r').read().split('\n')
	data_path = '../../data/Holmes_Training_Data'
	vocab_path = os.path.join(data_path, 'vocab.pkl')
	vocab_dict = pickle.load(open(vocab_path, 'rb'))
	vocab, vocab_id_map = vocab_dict['vocab'], vocab_dict['vocab_id_map']
	vocab_size = len(vocab)
	test_sentences = load_testing_data_with_multiple_option(path)
	# remove stop word
	# add start and end mark, map to id
	for i, test_sentence in enumerate(test_sentences):
		for j, sent in enumerate(test_sentence):
			clean_sent = clean_sentence(sent, stop_word)
			# add start and end mark
			clean_sent = '<START> ' + clean_sent + ' <END>'
			test_sentences[i][j] = embedded_sentence_by_id(clean_sent, vocab_id_map, n_step)

	hidden_size = 400
	n_layer = 1
	batch_size = 5
	num_sampled = 2000

	model = RNNModel(n_step, hidden_size, n_layer, batch_size, vocab_size, num_sampled)
	model_path = 'model'
	saver = tf.train.Saver()
	latest_checkpoint = tf.train.latest_checkpoint(model_path)
	with tf.Session() as sess:
		saver.restore(sess, latest_checkpoint)

		prediction = []
		option_list = ['a', 'b', 'c', 'd', 'e']
		for i, test_sentence in enumerate(test_sentences):
			if i > 0 and i % 10 == 0:
				print '{:.4f}% completed...'.format(i * 100. / len(test_sentences))
			stacked_sent = np.vstack(test_sentence)
			batch_x, batch_y = stacked_sent[:, :-1], stacked_sent[:, 1:]
			seq_loss_individual = sess.run(model.seq_loss_individual, feed_dict={model.x: batch_x, model.y: batch_y, model.keep_prob: 1.0})
			# seq_loss_individual = seq_loss_individual.reshape(batch_size, n_step-1)
			seq_loss_individual = seq_loss_individual.sum(axis=1)
			idx = np.argmin(seq_loss_individual)
			prediction.append(option_list[idx])
		prediction = np.array(prediction)

	answer_list = load_test_answer(os.path.join('../../data', 'test_answer.csv'))
	answer_list = np.array(answer_list)
	pickle.dump({'prediction': prediction, 'answer': answer_list}, open('prediction.pkl', 'wb'))
	print prediction
	print answer_list

	acc = (prediction.astype(str) == answer_list.astype(str)).sum() / float(len(answer_list))

	print 'accurary: {}'.format(acc)

if __name__ == '__main__':
	main()