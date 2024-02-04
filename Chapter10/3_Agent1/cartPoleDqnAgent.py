import collections
import random  # noqa: F401, RUF100
from typing import Any

import gym
import numpy as np

from cartPoleDqn import DQN


class Agent:
    def __init__(self, env: gym.Env) -> None:
        # DQN Env Variables
        self.env = env
        self.observations = self.env.observation_space.shape
        self.actions = self.env.action_space.n
        # DQN Agent Variables
        self.replay_buffer_size = 50_000
        self.train_start = 1_000
        self.memory: collections.deque = collections.deque(
            maxlen=self.replay_buffer_size
        )
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        # DQN Network Variables
        self.state_shape = self.observations
        self.learning_rate = 1e-3
        self.dqn = DQN(self.state_shape, self.actions, self.learning_rate)
        self.target_dqn = DQN(
            self.state_shape, self.actions, self.learning_rate
        )
        self.target_dqn.update_model(self.dqn)
        self.batch_size = 32

    def get_action(self, state: np.ndarray) -> Any:
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.actions)
        return np.argmax(self.dqn(state))

    def train(self, num_episodes: int) -> None:
        pass

    def remember(
        self,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        done: bool,
    ) -> None:
        pass

    def replay(self) -> None:
        pass

    def play(self, num_episodes: int, render: bool = True) -> None:
        pass


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = Agent(env)
