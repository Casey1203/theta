# coding: utf-8
# @Time    : 18-11-19 下午2:16
# @Author  : jia

from reader import load_holmes_raw_data
import os
import numpy as np

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


def embedded_sentence(sentence, stop_word, word_vector, word_vector_size):
	word_list = sentence.split()
	embedded_list = []
	for word in word_list:
		if word in word_vector and word not in stop_word:
			embedded_list.append(word_vector[word])
		else:
			embedded_list.append(np.zeros((word_vector_size,)))
	return embedded_list

def embedded_word(word, stop_word, word_vector, word_vector_size):
	if word in word_vector and word not in stop_word:
		return word_vector[word]
	else:
		return np.zeros((word_vector_size,))