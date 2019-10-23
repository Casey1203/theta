def get_prob(P, s, a, s1):
    return P(s, a, s1)

def get_reward(R, s, a):
    return R(s, a)

def display_dict(target_dict):
    for k in target_dict.keys():
        print('{}: {:.2f}'.format(k, target_dict[k]))
    print("")

def set_value(V, s, v):
    V[s] = v

def get_value(V, s):
    return V[s]

def display_V(V):
    for i in range(16):
        print('{0:>6.2f}'.format(V[i]), end=' ')
        if (i+1) % 4 == 0:
            print('')

def display_policy(policy, MDP, V):
    status_list, action_list, P, R, gamma = MDP
    for i in range(16):
        print('{0:^6}'.format(policy(MDP, V, status_list[i])), end=' ')
        if (i + 1) % 4 == 0:
            print('')
