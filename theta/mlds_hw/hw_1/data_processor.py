# coding: utf-8
import codecs
import re, os
import pickle
from gensim.models import word2vec, KeyedVectors
from collections import Counter
import numpy as np
import pandas as pd

WORD_VECTOR_SIZE = 300
MIN_WORD_COUNT = 20
STOP_WORD = open('../../data/stopwords.txt', 'r').read().split('\n')
OPTION_LIST = ['a)', 'b)', 'c)', 'd)', 'e)']
TRAINING_DATA_SIZE = 100000

def refine(data):
	# 只取出字符部分
	words = re.findall("[a-zA-Z'-]+", data)
	words = ["".join(word.split("'")) for word in words]
	data = ' '.join(words)
	return data

def get_testing_data_info(testing_data):
	max_test_length = 0
	min_test_length = float('inf')
	max_space_index = 0
	min_space_index = float('inf')
	sum_length = 0.
	sum_space_length = 0.
	for test_case in testing_data:
		sum_length += len(test_case['question'])
		if len(test_case['question']) > max_test_length:
			max_test_length = len(test_case['question'])
		if len(test_case['question']) < min_test_length:
			min_test_length = len(test_case['question'])
		space_index = test_case['question'].index('_____')
		sum_space_length += space_index
		if space_index > max_space_index:
			max_space_index = space_index
		if space_index < min_space_index:
			min_space_index = space_index
	average_space_index = sum_space_length / len(testing_data)
	average_length = sum_length / len(testing_data)
	seq_max_length = max_space_index + 1
	return max_test_length, min_test_length, average_length, max_space_index, min_space_index, average_space_index, seq_max_length


def process_training_and_testing_dataset():
	print 'load word vector model...'
	word_vector = KeyedVectors.load_word2vec_format('../../model/word_vector.bin', binary=True)
	training_data = [] # list of sentences
	all_words = "" # words in sentence
	# load training data
	print 'load and process training data...'
	for filename in os.listdir('../../data/Holmes_Training_Data'):
		with codecs.open('../../data/Holmes_Training_Data/%s' % filename, 'r', encoding='utf-8', errors='ignore') as f:
			print 'process %s' % filename
			parse_flag = False
			data = ""
			for line in f:
				# 在*END*THE SMALL PRINT!以上的文本内容没有用，丢弃
				if line.find("*END*THE SMALL PRINT!") > -1 or line.find("ENDTHE SMALL PRINT!") > -1:
					parse_flag = True
					continue
				if parse_flag:
					line = line.replace('\r\n', " ") # 句子末尾去掉回车等标示符
					data += line.lower()
			sentences = data.split('.') # 以.为分隔符划分句子
			training_data += [refine(sentence).split() for sentence in sentences]
			all_words += refine(data)

	# load test data
	print 'load and process testing data...'
	testing_word = ""
	testing_data_df = pd.read_csv('../../data/testing_data.csv', encoding='utf-8')
	testing_data = testing_data_df.to_dict(orient='records')
	for i in xrange(len(testing_data)):
		print 'process no. %s testing case, %s to go...' % (i, len(testing_data) - i - 1)
		testing_data[i]['question'] = [refine(word) if word != '_____' else '_____' for word in testing_data[i]['question'].lower().split()]
		testing_word += refine(' '.join(testing_data[i]['question'])).lower() + ' '
		for option in OPTION_LIST:
			testing_data[i][option] = refine(testing_data[i][option])
			testing_word += testing_data[i][option].lower() + ' '
	testing_word = testing_word[:-1]
	print 'get testing data info...'
	max_test_length, min_test_length, average_length, max_space_index, min_space_index, average_space_index, seq_max_length = \
		get_testing_data_info(testing_data)
	# embedding testing data
	print 'embedding testing data...'
	embedding_test_X = []
	embedding_test_Y = []
	for test_case in testing_data:
		pre_space_vector, post_space_vector, option_vector_dict = build_test_case_word_embedding(test_case, word_vector)
		for option in OPTION_LIST:
			embedding_test_X.append([pre_space_vector, option_vector_dict[option]])
			embedding_test_Y.append(post_space_vector)
	dump_embedding_testing_data(embedding_test_X, embedding_test_Y)
	# embedding training data
	embedding_training_X = []
	embedding_training_Y = []
	print 'embedding training data...'
	counter = 1
	post_fix = -1
	for sentence in training_data:
		valid_sentence_flag, pre_context_vector, post_context_vector, mid_context_vector = build_training_sentence_embedding(sentence, word_vector, max_test_length, min_test_length)
		if valid_sentence_flag:
			embedding_training_X.append([pre_context_vector, mid_context_vector])
			embedding_training_Y.append(post_context_vector)
			counter += 1
		if counter % TRAINING_DATA_SIZE == 0:
			counter = 1
			post_fix += 1
			dump_embedding_training_data(embedding_training_X, embedding_training_Y, post_fix)
			embedding_training_X = []
			embedding_training_Y = []
	return embedding_training_X, embedding_training_Y, embedding_test_X, embedding_test_Y


def build_test_case_word_embedding(test_case, word_vector):
	sentence = test_case['question']
	space_index = sentence.index('_____')
	pre_space_vector_arr = []
	for word in sentence[:space_index]:
		if word in word_vector and word not in STOP_WORD:
			pre_space_vector_arr.append(word_vector[word])
		else:
			pre_space_vector_arr.append(np.zeros((WORD_VECTOR_SIZE,)))
	pre_space_vector = np.sum(pre_space_vector_arr, axis=0)
	post_space_vector_arr = []
	for word in sentence[space_index+1:]:
		if word in word_vector and word not in STOP_WORD:
			post_space_vector_arr.append(word_vector[word])
		else:
			post_space_vector_arr.append(np.zeros((WORD_VECTOR_SIZE,)))
	post_space_vector = np.sum(post_space_vector_arr, axis=0)
	option_vector_dict = {}
	for key in OPTION_LIST:
		if test_case[key] in word_vector:
			option_vector_dict[key] = word_vector[test_case[key]]
		else:
			option_vector_dict[key] = np.zeros((WORD_VECTOR_SIZE,))
	return pre_space_vector, post_space_vector, option_vector_dict


def build_training_sentence_embedding(sentence, word_vector, max_test_length, min_test_length):
	valid_sentence_flag = 0
	if max_test_length >= len(sentence) >= min_test_length:  # 训练数据的句子长度在测试句子的长度之中
		valid_sentence_flag = 1
		for word in sentence:
			if word not in word_vector:
				valid_sentence_flag = 0
				break
	if valid_sentence_flag == 1:
		pre_context_vector_arr = []
		mid_index = len(sentence) / 2
		# pre context
		for word in sentence[:mid_index]:
			if word not in STOP_WORD and word in word_vector:
				pre_context_vector_arr.append(word_vector[word])
			else:
				pre_context_vector_arr.append(np.zeros((WORD_VECTOR_SIZE,)))
		pre_context_vector = np.sum(pre_context_vector_arr, axis=0)
		# post context
		post_context_vector_arr = []
		for word in sentence[mid_index + 1:]:
			if word not in STOP_WORD and word in word_vector:
				post_context_vector_arr.append(word_vector[word])
			else:
				post_context_vector_arr.append(np.zeros((WORD_VECTOR_SIZE,)))
		post_context_vector = np.sum(post_context_vector_arr, axis=0)
		if sentence[mid_index] in word_vector:
			mid_context_vector = word_vector[sentence[mid_index]]
		else:
			mid_context_vector = np.zeros((WORD_VECTOR_SIZE,))
		return valid_sentence_flag, pre_context_vector, post_context_vector, mid_context_vector
	else:
		return 0, None, None, None


def dump_embedding_data(embedding_train_X, embedding_train_Y, embedding_test_X, embedding_test_Y):

	embedding_test_data = {'embedding_test_X': embedding_test_X, 'embedding_test_Y': embedding_test_Y}
	for i in xrange(len(embedding_train_X) / TRAINING_DATA_SIZE):
		print 'dump embedding training data, postfix %s...' % i
		save_index = TRAINING_DATA_SIZE * i
		embedding_train_data = {
			'embedding_train_X': embedding_train_X[save_index: save_index+TRAINING_DATA_SIZE],
			'embedding_train_Y': embedding_train_Y[save_index: save_index+TRAINING_DATA_SIZE]}

		pickle.dump(embedding_train_data, open('../../data/embedding_train_data_%s.pkl' % i, 'wb'))
	print 'dump embedding testing data...'
	pickle.dump(embedding_test_data, open('../../data/embedding_test_data.pkl', 'wb'))


def dump_embedding_testing_data(embedding_test_X, embedding_test_Y):

	embedding_test_data = {'embedding_test_X': embedding_test_X, 'embedding_test_Y': embedding_test_Y}
	print 'dump embedding testing data...'
	pickle.dump(embedding_test_data, open('../../data/embedding_test_data.pkl', 'wb'))

def dump_embedding_training_data(embedding_train_X, embedding_train_Y, name):
	embedding_train_data = {
		'embedding_train_X': embedding_train_X,
		'embedding_train_Y': embedding_train_Y}
	print 'dump embedding training data...'
	pickle.dump(embedding_train_data, open('../../data/embedding_train_data_%s.pkl' % name, 'wb'))


if __name__ == '__main__':
	embedding_training_X, embedding_training_Y, embedding_test_X, embedding_test_Y = \
		process_training_and_testing_dataset()
