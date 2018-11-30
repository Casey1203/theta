# coding: utf-8
# @Time    : 18-11-21 下午6:17
# @Author  : jia

from reader import load_testing_data_and_split, load_test_answer
from rnn_1 import RNNModel
import pickle, os
import tensorflow as tf
import numpy as np
import pandas as pd
from parser import clean_sentence, embedded_sentence_by_id, test_sentence_info

def main():
	n_step = 20
	path = '../../data/testing_data.csv'
	stop_word = open('../../data/stopwords.txt', 'r').read().split('\n')
	data_path = '../../data/Holmes_Training_Data'
	vocab_path = os.path.join(data_path, 'vocab.pkl')
	vocab_dict = pickle.load(open(vocab_path, 'rb'))
	vocab, vocab_id_map = vocab_dict['vocab'], vocab_dict['vocab_id_map']
	vocab_size = len(vocab)
	test_sentences_forward, test_sentences_backward, option_dict_list = load_testing_data_and_split(path)

	# add start and end mark, map to id
	for i, test_sentence in enumerate(test_sentences_forward):
		clean_sent = clean_sentence(test_sentence, stop_word) # remove stop word
		clean_sent = '<START> ' + clean_sent
		test_sentences_forward[i] = embedded_sentence_by_id(clean_sent, vocab_id_map, n_step)
	for i, test_sentence in enumerate(test_sentences_backward):
		clean_sent = clean_sentence(test_sentence, stop_word) # remove stop word
		clean_sent += ' <END>' # add end mark
		test_sentences_backward[i] = embedded_sentence_by_id(clean_sent, vocab_id_map, n_step)



	hidden_size = 400
	n_layer = 1
	batch_size = 1
	num_sampled = 2000

	model = RNNModel(n_step, hidden_size, n_layer, batch_size, vocab_size, num_sampled, is_train=False)
	model_path = 'model'
	saver = tf.train.Saver()
	latest_checkpoint = tf.train.latest_checkpoint(model_path)
	with tf.Session() as sess:
		saver.restore(sess, latest_checkpoint)

		prediction = []
		option_list = ['a', 'b', 'c', 'd', 'e']
		for i in xrange(len(test_sentences_forward)):
			sent_forward = test_sentences_forward[i]
			sent_backward = test_sentences_backward[i]
			if i > 0 and i % 10 == 0:
				print '{:.4f}% completed...'.format(i * 100. / len(test_sentences_forward))
			probs = sess.run(model.probs, feed_dict={model.X_forward: [sent_forward], model.X_backward: [sent_backward], model.keep_prob: 1})
			probs_ser = pd.Series()
			for option in option_list:
				option_word = option_dict_list[i][option]
				if option_word in vocab:
					probs_ser[option] = probs[0][vocab_id_map[option_word]]
				else:
					probs_ser[option] = 0.
			max_option = probs_ser.argmax()
			prediction.append(max_option)
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