from random import shuffle
from queue import Queue
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

class Arena():
    def __init__(self, display=None, A=None):
        self.cards = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'] * 4
        self.cards_q = Queue(maxsize=52)
        self.cards_in_pool = [] # 用过的牌，公开
        self.display = display
        self.episodes = [] # 对局信息
        self.load_cards(self.cards)
        self.A = A

    def load_cards(self, cards):
        shuffle(cards)
        for card in cards:
            self.cards_q.put(card)
        cards.clear()

    def reward_of(self, dealer, player):
        dealer_point, _ = dealer.get_points()
        player_point, use_A = player.get_points()

        if player_point > 21: # dealer win
            reward = -1
        else:
            if player_point > dealer_point or dealer_point > 21: # player win
                reward = 1
            elif player_point == dealer_point: # draw game
                reward = 0
            else:
                reward = -1
        return reward, player_point, dealer_point, use_A

    def serve_card_to(self, player, n=1):
        """
        给Dealer和Player发牌，如果牌不够，则将pool中的牌重新洗后法
        :param player: Dealer or Player
        :param n: 发牌的数量
        :return:
        """

        cards = []
        for _ in range(n):
            # 从self.card_q中弹出
            if self.cards_q.empty():
                self._info('发牌器没牌了，重新整理废牌并洗牌')
                shuffle(self.cards_in_pool)
                self._info('整理了{}张废牌'.format(len(self.cards_in_pool)))
                assert len(self.cards_in_pool)>20 # 保证一次整理的牌数够多
                self.load_cards(self.cards_in_pool)
            cards.append(self.cards_q.get())
        self._info('发出了{}张牌{}给{}{}'.format(n, cards, player.role, player.name))

        player.receive(cards)
        player.cards_info()


    def _info(self, msg):
        if self.display:
            print(msg, end='\n')


    def recycle_cards(self, *players): # 玩家打出牌时，需要回收进pool
        if len(players) == 0:
            return
        for player in players:
            for card in player.cards:
                self.cards_in_pool.append(card)
            player.discharge_cards()

    def play_game(self, player, dealer):
        self._info('--------------开始新一局--------------')
        self.serve_card_to(player, n=2)
        self.serve_card_to(dealer, n=2)
        episode = []

        if player.policy is None:
            self._info('Player needs a policy')
            return
        if dealer.policy is None:
            self._info('Dealer needs a policy')
            return
        while True: # player start play!!
            action = player.policy() # stop or continue
            self._info("{}{} take action {}".format(player.role, player.name, action))

            player_state = player.get_state(dealer)
            episode.append((player_state, action)) # state, action pair
            if action == 'continue':
                self.serve_card_to(player)
            else:
                break
        reward, player_point, dealer_point, use_A = self.reward_of(dealer, player)
        if player_point > 21:
            # 游戏结束
            self._info('player{}爆点{}了，得分:{}'.format(player.name, player_point, reward))
            self.recycle_cards(player, dealer)
            self.episodes.append((episode, reward))
            self._info('--------------本局结束--------------')
            return episode, reward

        while True: # dealer start play!!
            action = dealer.policy()
            self._info("{}{} take action {}".format(dealer.role, dealer.name, action))
            if action == 'continue':
                self.serve_card_to(dealer)
            else:
                break
        reward, player_point, dealer_point, use_A = self.reward_of(dealer, player)
        player.cards_info()
        dealer.cards_info()

        if reward == 1:
            self._info('Player win')
        elif reward == -1:
            self._info('Player lose')
        else:
            self._info('Draw game')
        self._info('Player {}, Dealer {}'.format(player_point, dealer_point))
        self.recycle_cards(player, dealer)
        self.episodes.append((episode, reward))
        self._info('--------------本局结束--------------')
        return episode, reward

    def play_games(self, dealer, player, num=2, show_statistic=True):
        result = [0, 0, 0]
        self.episodes.clear()

        for i in tqdm(range(num)):
            episode, reward = self.play_game(player, dealer)
            result[1+reward] += 1
            if player.learning_method is not None:
                player.learning_method(episode, reward)
        if show_statistic:
            print('统计结果, play {} games, player win {}, lose {}, draw game {}, win ratio {}, not lose ratio {}'.format(
                num, result[2], result[0], result[1], result[2] / float(num),
                (result[2] +result[1]) / float(num)))


