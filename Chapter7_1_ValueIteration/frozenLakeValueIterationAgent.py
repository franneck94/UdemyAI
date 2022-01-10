from typing import Any
from typing import List
from typing import Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np


class Agent:
    def __init__(self, env: gym.Env) -> None:
        self.env = env
        self.observations: int = self.env.observation_space.shape[0]
        self.actions: int = self.env.action_space.n

    def get_action(self) -> Any:
        action = self.env.action_space.sample()
        return action

    def get_random_action(self) -> Any:
        action = self.env.action_space.sample()
        return action

    def get_samples(self, num_episodes: int) -> None:
        pass

    def compute_q_values(self) -> None:
        pass

    def train(self, num_iterations: int, num_epsiodes: int) -> None:
        pass

    def play(self, num_epsiodes: int, render: bool = True) -> None:
        for episode in range(num_epsiodes):
            state = self.env.reset()
            total_reward = 0.0

            while True:
                if render:
                    self.env.render()
                action = self.get_action()
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
                if done:
                    break

            print(f"Episode: {episode} Total Reward: {total_reward}")
        self.env.close()


def main() -> None:
    env = gym.make("FrozenLake-v1")
    agent = Agent(env)
    agent.train(num_iterations=10_000, num_epsiodes=1_000)
    agent.play(num_epsiodes=5)


if __name__ == "__main__":
    main()
