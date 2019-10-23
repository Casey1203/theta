from random import shuffle
from queue import Queue
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import numpy as np
from rl.model_free.utils import epsilon_greedy_policy


from rl.mdp.util import str_key, set_dict, get_dict


class Gamer():
    def __init__(self, name="", A=None, display=False):
        self.name = name
        self.cards = []
        self.display = display
        self.policy = None
        self.learning_method = None
        self.role = None
        self.A = A

    def __str__(self):
        return self.name

    def _value_of(self, card):
        try:
            card_num = int(card)
        except:
            if card in ['J', 'Q', 'K']:
                card_num = 10.
            elif card == 'A':
                card_num = 1
            else:
                card_num = 0
        return card_num

    def get_points(self):
        # 返回手牌中的点数和是否使用A
        total_point = 0
        num_use_A = 0
        if len(self.cards) == 0:
            return 0, bool(num_use_A)
        for card in self.cards:
            card_num = self._value_of(card)
            if card_num == 1:
                total_point += 11
                num_use_A += 1 # 使用次数加1
            else:
                total_point += card_num
        while total_point > 21 and num_use_A > 0: # 当点数超过21,且使用了A,则其中某些不能使用
            total_point -= 10
            num_use_A -= 1
        return total_point, bool(num_use_A)

    def receive(self, cards):
        assert isinstance(cards, list)
        for card in list(cards):
            self.cards.append(card)

    def discharge_cards(self): # 丢掉手牌
        self.cards.clear()

    def cards_info(self):
        self._info("{}{}现在的牌:{}".format(self.role, self.name, self.cards))

    def _info(self, msg):
        if self.display:
            print(msg, end='\n')


class Dealer(Gamer):
    def __init__(self, name="", A=None, display=False):
        super(Dealer, self).__init__(name, A, display)
        self.role = 'Dealer'
        self.policy = self.dealer_policy


    def first_card_value(self): # 因为第一张牌是明牌
        if self.cards is None or len(self.cards) == 0:
            return 0
        return self._value_of(self.cards[0])

    def dealer_policy(self):
        # 如果大于等于17点，则停止叫牌，否则继续叫牌
        dealer_point, _ = self.get_points()
        if dealer_point >= 17:
            return 'stop'
        else:
            return 'continue'


class Player(Gamer):
    def __init__(self, name='', A=None, display=None):
        super(Player, self).__init__(name, A, display)
        self.role = 'Player'
        self.policy = self.player_policy

    def player_policy(self):
        # 如果大于等于20点，则停止叫牌，否则继续叫牌
        dealer_point, _ = self.get_points()
        if dealer_point >= 20:
            return 'stop'
        else:
            return 'continue'

    def get_state(self, dealer):
        dealer_first_card_value = dealer.first_card_value()
        total_point, use_A = self.get_points()
        return dealer_first_card_value, total_point, use_A

    def get_state_name(self, dealer):
        str_key(self.get_state(dealer))


class MC_Player(Player):
    def __init__(self):
        super(MC_Player, self).__init__(name='', A=None, display=False)
        self.Q = {}
        self.Nsa = {}
        self.total_learning_times = 0
        self.policy = self.epsilon_greedy_policy
        self.learning_method = self.learn_Q

    def learn_Q(self, episode, r):
        """
        更新一次Q value
        :param episode: 一个状态
        :param r: G
        :return:
        """
        for s, a in episode:
            nsa = get_dict(self.Nsa, s, a)
            nsa += 1
            set_dict(self.Nsa, s, a)
            q = get_dict(self.Q, s, a)
            q = q + 1./nsa * (r - q)
            set_dict(self.Q, q, s, a)
        self.total_learning_times += 1

    def epsilon_greedy_policy(self, dealer, epsilon=None):
        player_points, _ = self.get_points()
        if player_points >= 21:
            return 'stop'
        if player_points < 12:
            return 'continue'
        else:
            A, Q = self.A, self.Q
            s = self.get_state_name(dealer)
            if epsilon is None:
                epsilon = 1.0 / (1+4 * math.log10(1+self.total_learning_times))
                return epsilon_greedy_policy(A, s, Q, epsilon)