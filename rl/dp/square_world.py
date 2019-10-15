from rl.dp.util import *
import numpy as np

def dynamics(s, a):
    """
    环境动力学，在状态s下，采取action a，得到reward，同时进入状态s'
    :param s:
    :param a:
    :return:
    """
    if (s % 4 == 0 and a == 'w') or (s < 4 and a == 'n') \
        or (s % 4 == 3 and a == 'e') or (s > 11 and a == 's') or s in [0, 15]:
        # 在墙角或墙壁
        s_prime = s
    else:
        ds = ds_actions[a]
        s_prime = s + ds
    reward = 0 if s in [0, 15] else -1
    is_end = True if s in [0, 15] else False
    return s_prime, reward, is_end

def P(s, a, s_prime): # status prob
    s_prime_dym, _, _ = dynamics(s, a)
    return s_prime == s_prime_dym

def R(s, a): # reward
    _, r, _ = dynamics(s, a)
    return r


def uniform_random_pi(MDP, V, s, a):
    _, action_list, _, _, _ = MDP
    n = len(action_list)
    return 0 if n == 0 else 1.0 / n

def greedy_pi(MDP, V, s, a):
    """
    计算行为的概率，该行为使得后续状态的价值最大
    :param MDP:
    :param V:
    :param s:
    :param a:
    :return:
    """
    status_list, action_list, R, P, gamma = MDP
    max_v = -np.inf
    max_v_action = []
    for action in action_list:
        s_prime, reward, _ = dynamics(s, action)
        v_s_prime = get_value(V, s_prime) # 后续状态的价值
        if v_s_prime > max_v:
            max_v = v_s_prime
            max_v_action = [action]
        elif v_s_prime == max_v:
            max_v_action.append(action)
    n = len(max_v_action)
    if n == 0:
        return 0.
    elif a in max_v_action:
        return 1./n
    else:
        return 0


def get_pi(Pi, s, a, MDP, V):
    return Pi(MDP, V, s, a)


def compute_q(MDP, V, s, a):
    # q是关于s和a的函数
    status_list, action_list, R, P, gamma = MDP
    q_sa = 0
    for s_prime in status_list:
        q_sa += get_prob(P, s, a, s_prime) * gamma * get_value(V, s_prime)
    q_sa += get_reward(R, s, a)
    return q_sa

def compute_v(MDP, V, Pi, s):
    # v是关于s的函数
    status_list, action_list, R, P, gamma = MDP
    v_s = 0
    for a in action_list:
        v_s += get_pi(Pi, s, a, MDP, V) * compute_q(MDP, V, s, a)
    return v_s


def update_V(MDP, V, Pi):
    status_list, _, _, _, _ = MDP
    V_prime = V.copy()
    for s in status_list:
        set_value(V_prime, s, compute_v(MDP, V_prime, Pi, s))
    return V_prime

def policy_evaluate(MDP, V, Pi, n):
    for i in range(n):
        V = update_V(MDP, V, Pi)
    return V

def policy_iterate(MDP, V, Pi, n, m):
    for i in range(m):
        V = policy_evaluate(MDP, V, Pi, n)
        Pi = greedy_pi
    return V


def compute_v_from_max_q(MDP, V, s):
    status_list, action_list, R, P, gamma = MDP
    v_s = -np.inf
    for a in action_list:
        qsa = compute_q(MDP, V, s, a)
        if qsa > v_s:
            v_s = qsa
    return v_s

def update_V_without_Pi(MDP, V):
    status_list, _, _, _, _ = MDP
    V_prime = V.copy()
    for s in status_list:
        v = compute_v_from_max_q(MDP, V_prime, s)
        set_value(V_prime, s, v)
    return V_prime

def value_iterate(MDP, V, n):
    for i in range(n):
        V = update_V_without_Pi(MDP, V)
    return V


def greedy_policy(MDP, V, s):
    status_list, action_list, R, P, gamma = MDP
    max_v = -np.inf
    max_v_action = []
    for action in action_list:
        s_prime, reward, _ = dynamics(s, action)
        v_s_prime = get_value(V, s_prime)
        if v_s_prime > max_v:
            max_v = v_s_prime
            max_v_action = [action]
        elif v_s_prime == max_v:
            max_v_action.append(action)
    return ''.join(max_v_action)



if __name__ == '__main__':
    V = [0 for _ in range(16)]

    status_list = [i for i in range(16)]
    action_list = ['n', 'e', 's', 'w']

    ds_actions = {'n': -4, 'e': 1, 'w': -1, 's': +4}

    gamma = 1.0

    MDP = status_list, action_list, R, P, gamma

    # V_pi = policy_iterate(MDP, V, greedy_pi, 100, 100)
    # display_V(V_pi)
    # V_star = value_iterate(MDP, V, 4)
    # display_V(V_star)

    display_policy(greedy_policy, MDP, V_star)