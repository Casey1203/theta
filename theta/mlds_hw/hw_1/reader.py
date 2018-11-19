# coding: utf-8
# @Time    : 18-11-15 下午1:07
# @Author  : jia
import codecs
import re
import pandas as pd

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

def load_testing_data(path):
	word = ""
	testing_data_df = pd.read_csv('../../data/testing_data.csv', encoding='utf-8')
	testing_data = testing_data_df.to_dict(orient='records')
	for i in xrange(len(testing_data)):
		testing_data[i]['question'] = [refine(word) if word != '_____' else '_____' for word in testing_data[i]['question'].lower().split()]
		testing_word += refine(' '.join(testing_data[i]['question'])).lower() + ' '
		for option in OPTION_LIST:
			testing_data[i][option] = refine(testing_data[i][option])
			testing_word += testing_data[i][option].lower() + ' '
	testing_word = testing_word[:-1]


if __name__ == '__main__':
	f = load_holmes_raw_data('../../data/Holmes_Training_Data/1ADAM10.TXT')
	pass
