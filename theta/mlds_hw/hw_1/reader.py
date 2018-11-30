# coding: utf-8
# @Time    : 18-11-15 下午1:07
# @Author  : jia
import codecs
import re
import pandas as pd

OPTION_LIST = ['a)', 'b)', 'c)', 'd)', 'e)']

def refine(data):
	# 只取出字符部分
	words = re.findall("[a-zA-Z'-]+", data)
	words = ["".join(word.split("'")) for word in words]
	data = ' '.join(words)
	return data

def load_holmes_raw_data(path):
	parse_mark = False
	mark1 = "*END*THE SMALL PRINT!"
	mark2 = "ENDTHE SMALL PRINT!"
	file_string = ""
	with codecs.open(path, 'r', encoding='utf-8', errors='ignore') as f:
		for line in f:
			if line.find(mark1) > -1 or line.find(mark2) > -1:
				parse_mark = True
				continue
			if parse_mark:
				line = line.replace('\r\n', " ")
				line = line.lower()
				file_string += line
		sentence_list = file_string.split('.')
		sentence_list = [refine(sentence) for sentence in sentence_list if len(refine(sentence)) > 0]
		word = refine(file_string)
	return sentence_list, word

def load_testing_data_with_multiple_option(path):
	testing_data_df = pd.read_csv(path, encoding='utf-8')
	testing_data = testing_data_df.to_dict(orient='records')
	OPTION_LIST = ['a)', 'b)', 'c)', 'd)', 'e)']
	test_sentences = []
	for i in xrange(len(testing_data)):
		testing_data[i]['question'] = ' '.join([refine(word) if word != '_____' else '_____' for word in testing_data[i]['question'].lower().split()])
		test_sentence_list = []
		for option in OPTION_LIST:
			test_sentence_list.append(testing_data[i]['question'].replace('_____', testing_data[i][option]))
		test_sentences.append(test_sentence_list)
	return test_sentences

def load_testing_data_and_split(path):
	testing_data_df = pd.read_csv(path, encoding='utf-8')
	testing_data = testing_data_df.to_dict(orient='records')
	OPTION_LIST = ['a)', 'b)', 'c)', 'd)', 'e)']
	test_sentences_forward = []
	test_sentences_backward = []
	option_dict_list = []
	for i in xrange(len(testing_data)):
		word_list = [refine(word) if word != '_____' else '_____' for word in testing_data[i]['question'].lower().split()]
		space_index = word_list.index('_____')
		test_sentences_forward.append(' '.join(word_list[:space_index]))
		test_sentences_backward.append(' '.join(word_list[space_index+1:]))
		option_dict = {}
		for option in OPTION_LIST:
			option_dict[option.replace(')', '')] = testing_data[i][option]
		option_dict_list.append(option_dict)
	return test_sentences_forward, test_sentences_backward, option_dict_list

def load_test_answer(path):
	test_answer_df = pd.read_csv(path, encoding='utf-8')
	return test_answer_df['answer'].tolist()

if __name__ == '__main__':
	f = load_holmes_raw_data('../../data/Holmes_Training_Data/1ADAM10.TXT')
	pass
