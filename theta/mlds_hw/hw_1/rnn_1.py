# coding: utf-8
# @Time    : 18-11-15 上午10:59
# @Author  : jia

# 思路：
# 以空格为分界线，前面一部分句子跑一个lstm，取空格前最后一个state作为句子的vector
# 后面一部分句子跑一个lstm，取空格后面第一个state作为句子的vector
# 把两个vector拼接起来，接一层fc，加一个softmax，得到具体输出的词

