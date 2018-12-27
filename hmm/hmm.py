# coding: utf-8

import numpy as np
import pandas as pd
import time

DELTA = 0.001

class HMM:
    def __init__(self, pi, A, B, state_set, observ_set):
        # N：状态数
        # M：观测数
        self.pi = pi
        self.A = A # num_state * num_state
        self.B = B # num_state * num_observ
        self.num_state = self.A.shape[0]
        self.num_observ = self.B.shape[1]
        self.state_set = state_set
        self.observ_set = observ_set
        assert self.num_state == len(self.state_set)
        assert self.num_observ == len(self.observ_set)

    def forward_prob(self, seq):
        T = len(seq)
        alpha = np.zeros([self.num_state, T]) # 横坐标为状态数，纵坐标为时序长度, alpha[i,j]表示时刻j，状态位于i，且观测到o1-oj
        alpha[:, 0] = self.pi * self.B[:, self.observ_set.index(seq[0])]
        for t in xrange(1, T):
            for n in xrange(self.num_state):
                alpha[n, t] = alpha[:, t-1].dot(self.A[:, n]) * self.B[n, self.observ_set.index(seq[t])]
        prob = np.sum(alpha[:, T-1]) # 句子observation的概率
        # alpha_dict = {}
        # alpha_dict[0] = self.pi * self.B[seq[0]]
        # for t in xrange(1, T):
        #     alpha_dict[t] = alpha_dict[t-1].dot(self.A[:, seq[t]]) * self.B[seq[t]]
        return alpha, prob

    def backward_prob(self, seq):
        T = len(seq)
        beta = np.zeros([self.num_state, T])
        beta[:, T-1] = 1
        for t in reversed(xrange(0, T-1)):
            # beta_dict[t] = self.A[seq[t], :].dot(self.B[seq[t],:]) * beta_dict[t+1]
            for n in xrange(self.num_state):
                beta[n, t] = np.sum(self.A[n, :] * self.B[:, self.observ_set.index(seq[t+1])] * beta[:, t+1])
        prob = np.sum(self.pi * self.B[:, self.observ_set.index(seq[0])] * beta[:, 0])
        return beta, prob

    def calc_gamma(self, alpha, beta):
        gamma = alpha * beta # N * T
        gamma /= np.sum(gamma, axis=0)
        return gamma

    def calc_xi(self, seq, alpha, beta):
        T = len(seq)
        xi = np.zeros([self.num_state, self.num_state, T-1])
        for t in xrange(T-1):
            for i in xrange(self.num_state):
                for j in xrange(self.num_state):
                    # nom = alpha_t[t] * self.A[seq[t], :] * self.B[seq[t], :] * beta_t[t]
                    # xi_t = nom / nom.sum()
                    xi[i, j, t] = alpha[i, t] * self.A[i, j] * self.B[j, self.observ_set.index(seq[t+1])] * beta[j, t+1]
            xi[:, :, t] /= np.sum(xi[:, :, t])
        return xi

    def baum_welch(self, seq):
        T = len(seq)
        alpha, prob_forward = self.forward_prob(seq)
        beta, prob_backward = self.backward_prob(seq)
        gamma = self.calc_gamma(alpha, beta)
        xi = self.calc_xi(seq, alpha, beta)
        while True:
            self.pi = gamma[:, 0]
            for i in xrange(self.num_state):
                denom = gamma[i, :T-1].sum()
                for j in xrange(self.num_state):
                    num = xi[i, j, :T-1].sum()
                    self.A[i, j] = num / denom
            for j in xrange(self.num_state):
                denom = gamma[j, :].sum()
                for k in xrange(self.num_observ):
                    num = 0
                    for t in xrange(T):
                        if self.observ_set.index(seq[t]) == k:
                            num += gamma[j, t]
                    self.B[j, k] = num / denom
            alpha, cur_prob_forward = self.forward_prob(seq)
            beta, cur_prob_backward = self.backward_prob(seq)
            gamma = self.calc_gamma(alpha, beta)
            xi = self.calc_xi(seq, alpha, beta)
            delta = cur_prob_forward - prob_forward
            prob_forward = cur_prob_forward
            print 'current delta is:', delta
            if delta <= DELTA:
                break

    def viterbi(self, seq):
        T = len(seq)
        delta = np.zeros([self.num_state, T])
        psi = np.zeros([self.num_state, T], dtype=np.int)
        delta[:, 0] = self.pi * self.B[:, self.observ_set.index(seq[0])]
        psi[:, 0] = 0
        best_trace = np.zeros(T, dtype=np.int)
        for t in xrange(1, T):
            for i in xrange(self.num_state):
                delta_a = delta[:, t-1] * self.A[:, i]
                max_j = np.argmax(delta_a)
                delta[i, t] = delta_a[max_j] * self.B[i, self.observ_set.index(seq[t])]
                psi[i, t] = max_j
        best_trace[T-1] = np.argmax(delta[:, T-1])
        for t in reversed(xrange(T-1)):
            best_trace[t] = psi[best_trace[t+1], t+1]
        return best_trace



def problem10_1():
    Q = [1, 2, 3]
    V = ['红', '白']
    A = np.array([
        [0.5, 0.2, 0.3],
        [0.3, 0.5, 0.2],
        [0.2, 0.3, 0.5]])
    B = np.array([
        [0.5, 0.5],
        [0.4, 0.6],
        [0.7, 0.3]])
    PI = np.array([[0.2, 0.4, 0.4]])
    # O = ['红', '白', '红', '红', '白', '红', '白', '白']
    O = ['红', '白', '红', '白']  # 习题10.1的例子

    alpha, prob = HMM(PI, A, B, Q, V).forward_prob(O)
    print prob

def problem10_2():
    A = np.array([
        [0.5, 0.1, 0.4],
        [0.3, 0.5, 0.2],
        [0.2, 0.2, 0.6]])

    B = np.array([
        [0.5, 0.5],
        [0.4, 0.6],
        [0.7, 0.3]])
    PI = np.array([[0.2, 0.3, 0.5]])
    V = ['红', '白']
    Q = [1, 2, 3]

    O = ['红', '白', '红', '红', '白', '红', '白', '白']
    hmm = HMM(PI, A, B, Q, V)
    alpha, prob_forward = hmm.forward_prob(O)
    beta, prob_backward = hmm.backward_prob(O)
    print 'prob_forward', prob_forward, 'prob_backward', prob_backward
    gamma = hmm.calc_gamma(alpha, beta)

    print gamma[2, 3]

def problem10_3():
    Q = [1, 2, 3]
    V = ['红', '白']
    A = np.array([
        [0.5, 0.2, 0.3],
        [0.3, 0.5, 0.2],
        [0.2, 0.3, 0.5]])
    B = np.array([
        [0.5, 0.5],
        [0.4, 0.6],
        [0.7, 0.3]])
    PI = np.array([[0.2, 0.4, 0.4]])
    # O = ['红', '白', '红', '红', '白', '红', '白', '白']
    O = ['红', '白', '红', '白']  # 习题10.1的例子
    print HMM(PI, A, B, Q, V).viterbi(O)



if __name__ == '__main__':
    # problem10_1()
    # problem10_2()
    problem10_3()