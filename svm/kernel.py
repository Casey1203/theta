# coding: utf-8
# @Time    : 18-10-25 下午3:15
# @Author  : jia
import numpy as np

def poly_kernel(x, z, p):
	return ((x * z).sum() + 1) ** p

def rbf_kernel(x, z, sigma):
	return np.exp(-(np.sqrt(((x - z) ** 2).sum()) / (2 * sigma ** 2)))
