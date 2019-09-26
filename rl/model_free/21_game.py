from random import shuffle
from queue import Queue
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from rl.mdp.util import str_key, set_dict, get_dict


class Gamer():
    def __init__(self, name="", A=None, display=False):
        self.name = name
        self.cards = []
        self.display = display
        self.policy = None
        self.learning_method = None
        self.A = A

    def __str__(self):
        return self.name

    def _value_of(self, card):
        pass

    def get_points(self):
        pass

    def receive(self, cards=[]):
        pass

    def discharge_cards(self):
        pass

    def cards_info(self):
        pass

    def _info(self, msg):
        pass