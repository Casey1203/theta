from rl.mdp.util import *
import numpy as np

def compute_q(MDP, V, s, a):
    # q是关于s和a的函数
    status_list, action_list, reward_map, P, gamma = MDP
    q_sa = 0
    for s_prime in status_list:
        q_sa += get_prob(P, s, a, s_prime) * gamma * get_value(V, s_prime)
    q_sa += get_reward(reward_map, s, a)
    return q_sa


def compute_v(MDP, V, Pi, s):
    # v是关于s的函数
    status_list, action_list, reward_map, P, gamma = MDP
    v_s = 0
    for a in action_list:
        v_s += get_pi(Pi, s, a) * compute_q(MDP, V, s, a)
    return v_s


def update_V(MDP, V, Pi):
    status_list, _, _, _, _ = MDP
    V_prime = V.copy()
    for s in status_list:
        V_prime[str_key(s)] = compute_v(MDP, V_prime, Pi, s)
    return V_prime

def policy_evaluate(MDP, V, Pi, n):
    for i in range(n):
        V = update_V(MDP, V, Pi)
    return V


def compute_v_from_max_q(MDP, V, s):
    status_list, action_list, reward_map, P, gamma = MDP
    v_s = -np.float('inf')
    for a in action_list:
        q_s_a = compute_q(MDP, V, s, a)
        if q_s_a > v_s:
            v_s = q_s_a
    return v_s

def update_V_without_pi(MDP, V):
    status_list, _, _, _, _ = MDP
    V_prime = V.copy()
    for s in status_list:
        V_prime[str_key(s)] = compute_v_from_max_q(MDP, V, s)
    return V_prime

def value_iterate(MDP, V, n):
    for i in range(n):
        V = update_V_without_pi(MDP, V)
    return V


if __name__ == '__main__':

    status_list = ['浏览手机中', '第一节课', '第二节课', '第三节课', '休息中']
    action_list = ['浏览手机', '学习', '离开浏览', '泡吧', '退出学习']

    reward_map = {}
    P = {} # 状态在take action后转移到另外一个状态的概率
    Pi = {} # 策略，action在某个statu下的的概率分布

    gamma = 1.0


    set_prob(P, status_list[0], action_list[0], status_list[0])
    set_prob(P, status_list[0], action_list[2], status_list[1])
    set_prob(P, status_list[1], action_list[0], status_list[0])
    set_prob(P, status_list[1], action_list[1], status_list[2])
    set_prob(P, status_list[2], action_list[4], status_list[4])
    set_prob(P, status_list[2], action_list[1], status_list[3])
    set_prob(P, status_list[3], action_list[1], status_list[4])
    set_prob(P, status_list[3], action_list[3], status_list[1], p=0.2)
    set_prob(P, status_list[3], action_list[3], status_list[2], p=0.4)
    set_prob(P, status_list[3], action_list[3], status_list[3], p=0.4)

    set_reward(reward_map, status_list[0], action_list[0], -1)
    set_reward(reward_map, status_list[0], action_list[2], 0)
    set_reward(reward_map, status_list[1], action_list[0], -1)
    set_reward(reward_map, status_list[1], action_list[1], -2)
    set_reward(reward_map, status_list[2], action_list[4], 0)
    set_reward(reward_map, status_list[2], action_list[1], -2)
    set_reward(reward_map, status_list[3], action_list[1], 10)
    set_reward(reward_map, status_list[3], action_list[3], 1)

    set_pi(Pi, status_list[0], action_list[0], 0.5)
    set_pi(Pi, status_list[0], action_list[2], 0.5)
    set_pi(Pi, status_list[1], action_list[0], 0.5)
    set_pi(Pi, status_list[1], action_list[1], 0.5)
    set_pi(Pi, status_list[2], action_list[1], 0.5)
    set_pi(Pi, status_list[2], action_list[4], 0.5)
    set_pi(Pi, status_list[3], action_list[1], 0.5)
    set_pi(Pi, status_list[3], action_list[3], 0.5)


    # display_dict(P)

    # display_dict(reward_map)

    V = {}

    MDP = status_list, action_list, reward_map, P, gamma

    # V = policy_evaluate(MDP, V, Pi, 100)

    # display_dict(V)

    v = compute_v(MDP, V, Pi, '第三节课')
    print('第三节课的最终价值', v)

    V = value_iterate(MDP, V, 4)
    display_dict(V)

    s = '第三节课'
    a = '泡吧'
    q = compute_q(MDP, V, s, a)
    print('在%s状态下采取行为%s的最优价值%s' % (s, a, q))