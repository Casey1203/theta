# coding: utf-8

import numpy as np
import pandas as pd
import time

DELTA = 0.001

class HMM:
    def __init__(self, pi, A, B):
        # N：状态数
        # M：观测数
        self.pi = pi
        self.A = A # num_state * num_state
        self.B = B # num_state * num_observ
        self.num_state = self.A.shape[0]
        self.num_observ = self.B.shape[1]

    def _forward_prob(self, seq):
        T = len(seq)
        alpha = np.zeros([self.num_state, T]) # 横坐标为状态数，纵坐标为时序长度, alpha[i,j]表示时刻j，状态位于i，且观测到o1-oj
        alpha[:, 0] = self.pi * self.B[:, seq[0]]
        for t in xrange(1, T):
            for n in xrange(self.num_state):
                alpha[n, t] = alpha[:, t-1].dot(self.A[:, n]) * self.B[n, seq[t]]
        prob = np.sum(alpha[:, T-1]) # 句子observation的概率
        # alpha_dict = {}
        # alpha_dict[0] = self.pi * self.B[seq[0]]
        # for t in xrange(1, T):
        #     alpha_dict[t] = alpha_dict[t-1].dot(self.A[:, seq[t]]) * self.B[seq[t]]
        return alpha, prob

    def _backward_prob(self, seq):
        T = len(seq)
        beta = np.zeros([self.num_state, T])
        beta[:, T-1] = 1
        for t in reversed(xrange(0, T-1)):
            # beta_dict[t] = self.A[seq[t], :].dot(self.B[seq[t],:]) * beta_dict[t+1]
            for n in xrange(self.num_state):
                beta[n, t] = np.sum(self.A[n, :] * (self.B[:, t+1]) * beta[: t+1]
        prob = np.sum(beta[:, 0])
        return beta, prob

    def _calc_gamma(self, alpha, beta):
        gamma = alpha * beta # N * T
        gamma /= np.sum(gamma, axis=0)
        return gamma

    def _calc_xi(self, seq, alpha, beta):
        T = len(seq)
        xi = np.zeros([self.num_state, self.num_state, T-1])
        for t in xrange(T):
            for i in xrange(self.num_state):
                for j in xrange(self.num_state):
                    # nom = alpha_t[t] * self.A[seq[t], :] * self.B[seq[t], :] * beta_t[t]
                    # xi_t = nom / nom.sum()
                    xi[i, j, t] = alpha[i, t] * self.A[i, j] * self.B[j, seq[t+1]] * beta[j, t+1]
            xi[:, :, t] /= np.sum(np.sum(xi[:,:,t],1),0)
        return xi

    def baum_welch(self, seq):
        T = len(seq)
        alpha, prob_forward = self._forward_prob(seq)
        beta, prob_backward = self._backward_prob(seq)
        gamma = self._calc_gamma(alpha, beta)
        xi = self._calc_xi(seq, alpha, beta)
        while True:
            self.pi = gamma[:, 0]
            for i in xrange(self.num_state):
                denom = gamma[i, :T-1].sum()
                for j in xrange(self.num_state):
                    num = xi[i, j, :T-1].sum()
                    self.A[i, j] = num / denom
            for j in xrange(self.num_state):
                for k in xrange(self.num_observ):
                    num = 0
                    for t in xrange(T):
                        if seq[t] == k:
                            num += gamma[j, t]
                    denom = gamma[j, :].sum()
                    self.B[j, k] = num / denom
            alpha, cur_prob_forward = self._forward_prob(seq)
            beta, cur_prob_backward = self._backward_prob(seq)
            gamma = self._calc_gamma(alpha, beta)
            xi = self._calc_xi(seq, alpha, beta)
            delta = cur_prob_forward - prob_forward
            prob_forward = cur_prob_forward
            print 'current delta is:', delta
            if delta <= DELTA:
                break
