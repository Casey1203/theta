# coding: utf-8
# @Time    : 18-10-24 下午3:00
# @Author  : jia

import numpy as np
from theta.util.file_loader import load_dataset

def get_support_vector(alphas, data_input, label_class):
	for i in xrange(len(alphas)):
		if alphas[i, 0] > 0:
			print '(%s, %s)' % (data_input[i], label_class[i])

def select_j_randomly(i, m):
	j = i
	while j == i:
		j = int(np.random.uniform(0, m))
	return j

def clip_alpha(alpha, H, L):
	if alpha > H:
		return H
	elif alpha < L:
		return L
	else:
		return alpha

def smo_simple(data_input, class_label, C, toler, maxIter):
	# init
	b = 0
	m, n = np.shape(data_input)
	alphas = np.zeros((m, 1))
	iteration = 0
	while iteration < maxIter:
		alpha_pair_change = 0
		for i in xrange(m):
			gx_i = ((alphas * class_label).reshape(m, ) * (data_input[i, :] * data_input).sum(axis=1)).sum() + b
			E_i = gx_i - class_label[i, 0]

			if (class_label[i, 0] * E_i < -toler and alphas[i, 0] < C) or (class_label[i, 0] * E_i > toler and alphas[i, 0] > 0):
				j = select_j_randomly(i, m)
				gx_j = ((alphas * class_label).reshape(m, ) * (data_input[j, :] * data_input).sum(axis=1)).sum() + b
				E_j = gx_j - class_label[j, 0]
				alpha_i_old = alphas[i, 0].copy()
				alpha_j_old = alphas[j, 0].copy()

				if class_label[i, 0] != class_label[j, 0]:
					L = max(0, alpha_j_old - alpha_i_old)
					H = min(C, C + alpha_j_old - alpha_i_old)
				else:
					L = max(0, alpha_j_old + alpha_i_old - C)
					H = min(C, alpha_j_old + alpha_i_old)
				if L == H:
					print 'L == H'
					continue
				eta = (data_input[i, :] ** 2).sum() + (data_input[j, :] ** 2).sum() - 2 * (data_input[i, :] * data_input[j, :]).sum()
				alphas[j, 0] += class_label[j, 0] * (E_i - E_j) / eta
				alphas[j, 0] = clip_alpha(alphas[j, 0], H, L)
				if abs(alphas[j, 0] - alpha_j_old) < 0.00001:
					print 'j not moving enough'
					continue
				alphas[i, 0] += class_label[i, 0] * class_label[j, 0] * (alpha_j_old - alphas[j, 0])

				b_i_new = -E_i - class_label[i, 0] * (data_input[i, :] * data_input[i, :]).sum() * (alphas[i, 0] - alpha_i_old) - \
					class_label[j, 0] * (data_input[j, :] * data_input[i, :]).sum() * (alphas[j, 0] - alpha_j_old) + b
				b_j_new = -E_j - class_label[i, 0] * (data_input[i, :] * data_input[j, :]).sum() * (alphas[i, 0] - alpha_i_old) - \
					class_label[j, 0] * (data_input[j, :] * data_input[j, :]).sum() * (alphas[j, 0] - alpha_j_old) + b
				if 0 < alphas[i, 0] < C:
					b = b_i_new
				elif 0 < alphas[j, 0] < C:
					b = b_j_new
				else:
					b = (b_i_new + b_j_new) / 2

				alpha_pair_change += 1
				print 'iter: %s, i: %s, pair change %s' % (iteration, i, alpha_pair_change)
		if alpha_pair_change == 0:
			iteration += 1
		else:
			iteration = 0
		print 'iteration number: %s' % iteration
	return b, alphas

if __name__ == '__main__':
	data_path = '../data/testSet.txt'
	data_arr, label_arr = load_dataset(data_path)
	b, alphas = smo_simple(data_arr, label_arr, 0.6, 0.001, 40)
	print 'b:', b
	print 'alphas:', alphas
	get_support_vector(alphas, data_arr, label_arr)