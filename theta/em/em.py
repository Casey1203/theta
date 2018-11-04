# coding: utf-8
import numpy as np


def get_mu(y, pi, p, q):
	num_b = pi * p ** y * (1 - p) ** (1 - y)
	num_c = (1 - pi) * q ** y * (1 - q) ** (1 - y)
	new_mu = num_b / (num_b + num_c)
	return new_mu


def update_param(y_arr, pi, p, q):
	y_len = len(y_arr)
	mu_arr = np.ones((y_len,))

	for i in xrange(y_len):
		mu_arr[i] = get_mu(y_arr[i], pi, p, q)
	new_pi = mu_arr.sum() / y_len
	new_p = mu_arr.dot(y_arr) / mu_arr.sum()
	new_q = (1 - mu_arr).dot(y_arr) / (1 - mu_arr).sum()

	return new_pi, new_p, new_q

if __name__ == '__main__':
	y_arr = np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 1])
	pi, p, q = 0.4, 0.6, 0.7
	print update_param(y_arr, pi, p, q)
