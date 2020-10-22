import collections
import random

import gym
import numpy as np

from cartPoleDqn import DQN


class Agent:
    def __init__(self, env: gym.Env):
        # DQN Env Variables
        self.env = env
        self.observations = self.env.obersvation_space.shape
        self.actions = self.env.action_space.n
        # DQN Agent Variables
        self.replay_buffer_size = 50_000
        self.train_start = 1_000
        self.memory = collections.deque(maxlen=self.replay_buffer_size)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        # DQN Network Variables
        self.state_shape = self.observations
        self.learning_rate = 1e-3
        self.dqn = DQN(
            self.state_shape,
            self.actions,
            self.learning_rate
        )
        self.target_dqn = DQN(
            self.state_shape,
            self.actions,
            self.learning_rate
        )
        self.target_dqn.update_model(self.dqn)
        self.batch_size = 32

    def get_action(self, state: np.ndarray):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.actions)
        else:
            return np.argmax(self.dqn(state))

    def train(self, num_episodes: int):
        pass

    def remember(self, state, action, reward, next_state, done):
        pass

    def replay(self):
        pass

    def play(self, num_episodes: int, render: bool = True):
        pass


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = Agent(env)
