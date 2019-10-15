from random import shuffle
from queue import Queue
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import numpy as np
from rl.mdp.util import *
from rl.model_free.arena import Arena
from rl.model_free.role import *

from mpl_toolkits.mplot3d import Axes3D

def policy_evaluate(episoldes, V, Ns):
    """
    根据MC sample得到的Ns个episode，通过平均的方法，计算每个状态s的value
    :param episoldes: 多个episode
    :param V:
    :param Ns: number of episode
    :return:
    """
    for episode, r in episoldes:
        for s, a in episode:
            ns = get_dict(Ns, s) # 状态的访问次数
            v = get_dict(V, s) # 状态的价值
            set_dict(Ns, ns+1, s)
            set_dict(V, v + (r-v)/(ns+1), s)


def draw_value(value_dict, useable_ace=True, is_q_dict=False, A = None):
    fig = plt.figure()
    ax = Axes3D(fig)

    x = np.arange(1, 11, 1)
    y = np.arange(12, 22, 1)

    X, Y = np.meshgrid(x, y)

    row, col = X.shape

    Z = np.zeros((row, col))
    if is_q_dict:
        n = len(A)

    for i in range(row):
        for j in range(col):
            # state_name = str(X[i, j]) + '_' + str(Y[i, j]) + '_' + str(useable_ace)
            state_name = (X[i, j], Y[i, j], useable_ace)
            if not is_q_dict:
                Z[i, j] = get_dict(value_dict, state_name)
            else:
                assert (A is not None)
                for a in A:
                    new_state_name = state_name + '_' + str(a)
                    q = get_dict(value_dict, new_state_name)
                    if q >= Z[i, j]:
                        Z[i, j] = q
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color='lightgray')
    plt.show()




if __name__ == '__main__':
    A = ['continue', 'stop']
    display=False

    player = Player(A=A, display=display)
    dealer = Dealer(A=A, display=display)
    arena = Arena(A=A, display=display)
    arena.play_games(dealer, player, num=200000)


    V = {}
    Ns = {}

    policy_evaluate(arena.episodes, V, Ns)

    # display_dict(V)


    draw_value(V, True, A=A)