# coding: utf-8
# @Time    : 18-10-24 下午3:00
# @Author  : jia

import numpy as np
from theta.util.file_loader import load_dataset

# 获得支撑向量
def get_support_vector_index(alphas):
	support_vector_index = []
	for i in xrange(len(alphas)):
		if alphas[i, 0] > 0:
			support_vector_index.append(i)
	return support_vector_index

def get_support_vector(support_vector_index, data_input, label_class):
	for i in support_vector_index:
		print '(%s, %s)' % (data_input[i], label_class[i])


def get_non_bound_alpha(alphas, C):
	no_bound_index = []
	for i in xrange(len(alphas)):
		if C > alphas[i, 0] > 0:
			no_bound_index.append(i)
	return no_bound_index

# 随机选择下标
def select_j_randomly(i, m):
	j = i
	while j == i:
		j = int(np.random.uniform(0, m))
	return j

# 剪辑alpha
def clip_alpha(alpha, H, L):
	if alpha > H:
		return H
	elif alpha < L:
		return L
	else:
		return alpha

# 随机选择j的情况下的smo算法
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
				if eta <= 0:
					print 'eta <= 0'
					continue
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


class OptStruct:
	def __init__(self, data_input, class_label, C, toler):
		self.data_input = data_input
		self.class_label = class_label
		self.C = C
		self.toler = toler
		self.m = np.shape(data_input)[0]
		self.alphas = np.zeros((self.m, 1)) # init alpha
		self.b = 0 # init b
		self.e_cache = np.zeros((self.m, 2)) # 第一列表示valid，第二列表示E值

# 计算E值
def calc_E_k(opt_struct, k):
	gx_k = ((opt_struct.alphas * opt_struct.class_label).reshape(opt_struct.m, ) * (opt_struct.data_input[k, :] * opt_struct.data_input).sum(axis=1)).sum() + opt_struct.b
	E_k = gx_k - opt_struct.class_label[k, 0]
	return E_k

# 根据Ei和Ej的差最大的原则选择j
def select_j(opt_struct, i, E_i):
	opt_struct.e_cache[i] = [1, E_i] # set index i as [valid, E_i]
	valid_e_cahce_list = np.nonzero(opt_struct.e_cache[:, 0])[0]
	max_delta_E = -1
	max_k = -1
	E_j = 0
	if len(valid_e_cahce_list) > 1:
		for k in valid_e_cahce_list:
			if k == 1:
				continue
			E_k = calc_E_k(opt_struct, k) # 计算第k个数据点的E值
			delta_E = np.abs(E_i - E_k)
			if delta_E > max_delta_E:
				max_k = k
				max_delta_E = delta_E
				E_j = E_k
		return max_k, E_j # 返回deltaE最大的数据点对应的index和E值
	else: # 随机选
		j = select_j_randomly(i, opt_struct.m)
		E_j = calc_E_k(opt_struct, j)
		return j, E_j

# 更新E值表
def update_E_k(opt_struct, k):
	E_k = calc_E_k(opt_struct, k)
	opt_struct.e_cache[k] = [1, E_k]


def inner_loop(i, opt_struct):
	# i是外层循环选择的alpha下标
	E_i = calc_E_k(opt_struct, i)
	if (opt_struct.class_label[i, 0] * E_i < -opt_struct.toler and opt_struct.alphas[i, 0] < opt_struct.C) or \
		(opt_struct.class_label[i, 0] * E_i > opt_struct.toler and opt_struct.alphas[i, 0] > 0):
		j, E_j = select_j(opt_struct, i, E_i)
		alpha_i_old = opt_struct.alphas[i, 0].copy()
		alpha_j_old = opt_struct.alphas[j, 0].copy()

		if opt_struct.class_label[i, 0] != opt_struct.class_label[j, 0]:
			L = max(0, alpha_j_old - alpha_i_old)
			H = min(opt_struct.C, opt_struct.C + alpha_j_old - alpha_i_old)
		else:
			L = max(0, alpha_j_old + alpha_i_old - opt_struct.C)
			H = min(opt_struct.C, alpha_j_old + alpha_i_old)
		if L == H:
			print 'L == H'
			return 0

		eta = (opt_struct.data_input[i, :] ** 2).sum() + (opt_struct.data_input[j, :] ** 2).sum() - 2 * (opt_struct.data_input[i, :] * opt_struct.data_input[j, :]).sum()
		if eta <= 0:
			print 'eta <= 0'
			return 0
		opt_struct.alphas[j, 0] += opt_struct.class_label[j, 0] * (E_i - E_j) / eta
		opt_struct.alphas[j, 0] = clip_alpha(opt_struct.alphas[j, 0], H, L)
		update_E_k(opt_struct, j)
		if abs(opt_struct.alphas[j, 0] - alpha_j_old) < 0.00001:
			print 'j not moving enough'
			return 0
		opt_struct.alphas[i, 0] += opt_struct.class_label[i, 0] * opt_struct.class_label[j, 0] * (alpha_j_old - opt_struct.alphas[j, 0])
		update_E_k(opt_struct, i)
		b_i_new = -E_i - \
				opt_struct.class_label[i, 0] * (opt_struct.data_input[i, :] * opt_struct.data_input[i, :]).sum() * (opt_struct.alphas[i, 0] - alpha_i_old) - \
				opt_struct.class_label[j, 0] * (opt_struct.data_input[j, :] * opt_struct.data_input[i, :]).sum() * (opt_struct.alphas[j, 0] - alpha_j_old) + opt_struct.b
		b_j_new = -E_j - \
				opt_struct.class_label[i, 0] * (opt_struct.data_input[i, :] * opt_struct.data_input[j, :]).sum() * (opt_struct.alphas[i, 0] - alpha_i_old) - \
				opt_struct.class_label[j, 0] * (opt_struct.data_input[j, :] * opt_struct.data_input[j, :]).sum() * (opt_struct.alphas[j, 0] - alpha_j_old) + opt_struct.b
		if 0 < opt_struct.alphas[i, 0] < opt_struct.C:
			opt_struct.b = b_i_new
		elif 0 < opt_struct.alphas[j, 0] < opt_struct.C:
			opt_struct.b = b_j_new
		else:
			opt_struct.b = (b_i_new + b_j_new) / 2

		return 1
	else:
		return 0

def outer_loop(data_input, class_label, C, toler, maxIter):
	opt_struct = OptStruct(data_input, class_label, C, toler)
	iteration = 0
	entire_set = True
	alpha_pair_change = 0
	while iteration < maxIter and (alpha_pair_change > 0 or entire_set):
		# 当选择alpha1是在整个数据集上进行，且在这种情况下还没办法对alpha pair产生change，此时认为alpha已经无需调整了
		alpha_pair_change = 0
		if entire_set:
			for i in xrange(opt_struct.m):
				alpha_pair_change += inner_loop(i, opt_struct)
				print 'full_set, iter: %s, i: %s, pairs change %s' % (iteration, i, alpha_pair_change)
			iteration += 1
		else:
			data_no_bound = get_non_bound_alpha(opt_struct.alphas, opt_struct.C)
			for i in data_no_bound:
				alpha_pair_change += inner_loop(i, opt_struct)
				print 'no bound, iter %s, i: %s, pairs change %s' % (iteration, i, alpha_pair_change)
			iteration += 1
		if entire_set:
			entire_set = False
		elif alpha_pair_change == 0:
			entire_set = True
		print 'iter number: %s' % iteration
	return opt_struct.b, opt_struct.alphas

def calc_w(alphas, class_label, data_input):
	w = ((alphas * class_label) * data_input).sum(axis=0)
	return w

if __name__ == '__main__':
	data_path = '../data/testSet.txt'
	data_arr, label_arr = load_dataset(data_path)
	# simple smo
	# b, alphas = smo_simple(data_arr, label_arr, 0.6, 0.001, 40)
	# full smo
	b, alphas = outer_loop(data_arr, label_arr, 0.6, 0.001, 40)
	w = calc_w(alphas, label_arr, data_arr)
	print 'b:', b
	print 'alphas:', alphas
	print 'w:', w
	support_vector_index = get_support_vector_index(alphas)
	get_support_vector(support_vector_index, data_arr, label_arr)

	# test label
	print 'data_0: sign %s, label: %s' % (np.sign(data_arr[0, :].dot(w) + b), label_arr[0])
	print 'data_1: sign %s, label: %s' % (np.sign(data_arr[1, :].dot(w) + b), label_arr[1])
	print 'data_2: sign %s, label: %s' % (np.sign(data_arr[2, :].dot(w) + b), label_arr[2])