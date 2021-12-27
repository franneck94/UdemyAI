import gym
import numpy as np

from environment import Env
from environment import GraphicDisplay


class Agent:
    def __init__(self, env: gym.Env) -> None:
        self.env = env
        self.S = self.env.all_state
        self.A = self.env.possible_actions
        self.cols, self.rows = self.env.width, self.env.height
        self.num_actions = len(self.A)
        self.num_states = len(self.S)
        self.gamma = 0.9
        self.init_prob = 1.0 / self.num_actions
        self.policy = [
            [[self.init_prob for _ in range(self.num_actions)] for _ in range(self.cols)]
            for _ in range(self.rows)
        ]
        self.v_values = [[0.0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.policy[2][2] = [0.0 for _ in range(self.num_actions)]
