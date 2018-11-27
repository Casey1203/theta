# coding: utf-8
# @Time    : 18-11-19 下午2:16
# @Author  : jia

import os
import numpy as np
from collections import Counter

def sentence_parser(sentence, min_length, stop_word):
	word_list = sentence.split()
	word_list = [word for word in word_list if word not in stop_word]
	if len(word_list) < min_length:
		return 0,
	mid_index = len(word_list)
	forward_sentence = ' '.join(word_list[:mid_index])
	backward_sentence = ' '.join(word_list[mid_index+1:])
	mid_word = word_list[mid_index]
	return 1, forward_sentence, backward_sentence, mid_word

def clean_sentence(sentence, stop_word):
	word_list = sentence.split()
	word_list = [word for word in word_list if word not in stop_word]
	return ' '.join(word_list)

def embedded_sentence_by_word2vec(sentence, stop_word, word_vector, word_vector_size, min_length):
	word_list = sentence.split()
	word_list = [word for word in word_list if word not in stop_word]
	embedded_list = []
	if len(word_list) < min_length:
		return embedded_list
	for word in word_list:
		if word in word_vector and word not in stop_word:
			embedded_list.append(word_vector[word])
		else:
			embedded_list.append(np.zeros((word_vector_size,)))
	return embedded_list

def embedded_sentence_by_id(sentence, vocab_map_id, seq_length):
	word_list = sentence.split()
	id_s = np.ones(seq_length, np.int32) * vocab_map_id['<END>']
	for i, word in enumerate(word_list):
		id_s[i] = vocab_map_id.get(word, 0)
	return id_s

def build_vocab(sentences_list):
	words = []
	for sentence in sentences_list:
		words += sentence.split()
	word_counter = Counter(words)
	vocab = word_counter.keys()
	vocab = ['<UNK>'] + vocab

	vocab_id_map = {x: i for i, x in enumerate(vocab)}
	return vocab, vocab_id_map

def embedded_word_by_id(word, vocab_map_id):
	return vocab_map_id.get(word, 0)

def test_sentence_info(test_sentences):
	max_length = 0
	for i, test_sentence in enumerate(test_sentences):
		for j, sent in enumerate(test_sentence):
			if len(sent.split()) > max_length:
				max_length = len(sent.split())
				print 'max_length', max_length