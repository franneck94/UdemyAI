from typing import Any

import gym
import matplotlib.pyplot as plt  # noqa: F401, RUF100
import numpy as np
from keras.layers import Activation  # noqa: F401, RUF100
from keras.layers import Dense  # noqa: F401, RUF100
from keras.models import Sequential
from keras.optimizers import Adam  # noqa: F401, RUF100
from keras.utils import to_categorical  # noqa: F401, RUF100


class Agent:
    def __init__(self, env: gym.Env) -> None:
        self.env = env
        self.observations: int = self.env.observation_space.shape[0]
        self.actions: int = self.env.action_space.n

    def get_model(self) -> Sequential:
        pass

    def get_action(self) -> Any:
        return self.env.action_space.sample()

    def get_samples(self, num_episodes: int) -> tuple[list[float], list[float]]:
        pass

    def filter_episodes(
        self,
        rewards: list[float],
        episodes: list[tuple[float, float]],
        percentile: float,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        pass

    def train(
        self, percentile: float, num_iterations: int, num_episodes: int
    ) -> tuple[list[float], list[float]]:
        pass

    def play(self, episodes: int, render: bool = True) -> None:
        for episode in range(episodes):
            self.env.reset()
            total_reward = 0.0

            while True:
                if render:
                    self.env.render()
                action = self.get_action()
                _, reward, done, _ = self.env.step(action)
                total_reward += reward
                if done:
                    break

            print(f"Episode: {episode} Total Reward: {total_reward}")
        self.env.close()


def main() -> None:
    env = gym.make("CartPole-v1")
    agent = Agent(env)
    print(agent)


if __name__ == "__main__":
    main()
