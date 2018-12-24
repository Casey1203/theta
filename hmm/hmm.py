# coding: utf-8

import numpy as np
import pandas as pd
import time


class HMM:
    def __init__(self):
        # N：状态数
        # M：观测数
        self.pi = None # N x 1
        self.A = None # N x N
        self.B = None # N x M

    def _forward_prob(self, seq):
        T = len(seq)
        alpha_dict = {}
        alpha_dict[0] = self.pi * self.B[seq[0]]
        for t in xrange(1, T):
            alpha_dict[t] = alpha_dict[t-1].dot(self.A[:, seq[t]]) * self.B[seq[t]]
        return alpha_dict

    def _calculate_prob_forward_algo(self, seq):
        T = len(seq)
        alpha_dict = self._forward_prob(seq)
        return alpha_dict[T-1].sum()

    def _backward_prob(self, seq):
        T = len(seq)
        beta_dict = {}
        beta_dict[T] = np.ones(T,)
        for t in xrange(T-2, 0, -1):
            beta_dict[t] = self.A[seq[t], :].dot(self.B[seq[t],:]) * beta_dict[t+1]
        return beta_dict

    def _calculate_prob_backward_algo(self, seq):
        T = len(seq)
        beta_dict = self._backward_prob(seq)
        return (self.pi * self.B[:, seq[0]] * beta_dict[0]).sum()

    def calc_prob(self, seq):
        self._calculate_prob_forward_algo(seq)
        self._calculate_prob_backward_algo(seq)


    def _calc_gamma(self, seq, T, alpha_t, beta_t):
        for t in xrange(T):
            pass


    def learning_param(self, dataset, is_supervise=False):
        if is_supervise:
            x = dataset['x'] # 观测序列
            y = dataset['y'] # 状态序列
        else:
            x = dataset['x']  # 观测序列
            S = len(x)

