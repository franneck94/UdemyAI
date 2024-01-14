from typing import Any

import gym
import matplotlib.pyplot as plt  # noqa: F401, RUF100
import numpy as np  # noqa: F401, RUF100


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

    def train(self, num_iterations: int, num_episodes: int) -> None:
        pass

    def play(self, num_episodes: int, render: bool = True) -> None:
        for episode in range(num_episodes):
            _ = self.env.reset()
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
    env = gym.make("FrozenLake-v0")
    agent = Agent(env)
    agent.train(num_iterations=10_000, num_episodes=1_000)
    agent.play(num_episodes=5)


if __name__ == "__main__":
    main()
