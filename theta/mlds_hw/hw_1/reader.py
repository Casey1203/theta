# coding: utf-8
# @Time    : 18-11-15 下午1:07
# @Author  : jia
import codecs
import re

def refine(data):
	# 只取出字符部分
	words = re.findall("[a-zA-Z'-]+", data)
	words = ["".join(word.split("'")) for word in words]
	data = ' '.join(words)
	return data

def load_holmes_raw_data(train_path):
	parse_mark = False
	mark1 = "*END*THE SMALL PRINT!"
	mark2 = "ENDTHE SMALL PRINT!"
	file_string = ""
	with codecs.open(train_path, 'r', encoding='utf-8', errors='ignore') as f:
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


if __name__ == '__main__':
	f = load_holmes_raw_data('../../data/Holmes_Training_Data/1ADAM10.TXT')
	pass
