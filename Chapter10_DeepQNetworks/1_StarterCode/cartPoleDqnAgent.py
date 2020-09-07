import random
import collections

import gym
import numpy as np

from cartPoleDqn import *


class Agent:
    def __init__(self, env):
        # DQN Env Variables
        self.env = env
        self.observations = self.env.observation_space.n
        self.actions = self.env.action_space.n
        # DQN Agent Variables
        self.replay_buffer_size = 50000
        self.train_start = 1000
        self.memory = collections.deque(maxlen=self.replay_buffer_size)
        self.gamma = 0.95
        # DQN Network Variables

    def get_action(self):
        pass

    def train(self):
        pass

    def play(self):
        pass


if __name__ == "__main__":
    pass
