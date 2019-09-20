import numpy as np
import pandas as pd


def compute_return(start_index, chain, gamma, reward_map) -> float:
    retrn = 0.
    power = 0.
    for i in range(start_index, len(chain)):
        status = chain[i]
        reward = reward_map[status]
        retrn += reward * np.power(gamma, power)
        power += 1
    return retrn

def compute_value(P_df, reward_map, gamma):
    v = np.linalg.inv((np.identity(len(reward_map)) - gamma * P_df)).dot(reward_map.values)
    return v



if __name__ == '__main__':
    reward_list = [-2, -2, -2, 10, 1, -1, 0]
    status_list = ['C1', 'C2', 'C3', 'PASS', 'PUB', 'FB', 'SLEEP']

    reward_map = pd.Series(index=status_list, data=reward_list)

    chain = ['C1', 'FB', 'FB', 'C1', 'C2', 'C3', 'PUB', 'C1', 'FB', 'FB', 'FB', 'C1', 'C2', 'C3', 'PUB', 'C2', 'SLEEP']

    # retrn = compute_return(0, chain, 0.5, reward_map)
    # print(retrn)

    P_mat = [
        [0, 0.5, 0, 0, 0, 0.5, 0],
        [0, 0, 0.8, 0, 0, 0, 0.2],
        [0, 0, 0, 0.6, 0.4, 0, 0],
        [0, 0, 0, 0, 0, 0, 1.],
        [0.2, 0.4, 0.4, 0, 0, 0, 0],
        [0.1, 0, 0, 0, 0, 0.9, 0],
        [0, 0, 0, 0, 0, 0, 1]
    ]

    P_df = pd.DataFrame(index=status_list, columns=status_list, data=P_mat)
    v = compute_value(P_df, reward_map, gamma=0.999999999)
    print(v)


