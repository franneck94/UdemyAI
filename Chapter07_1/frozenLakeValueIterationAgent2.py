from typing import Any

import gym
import matplotlib.pyplot as plt
import numpy as np

from plotting import action_map
from plotting import plotting_fn


class Agent:
    def __init__(self, env: gym.Env) -> None:
        self.env = env
        self.observations: int = self.env.observation_space.shape[0]
        self.actions: int = self.env.action_space.n
        self.gamma = 0.9
        self.state = self.env.reset()
        self.S = range(self.observations)
        self.A = range(self.actions)
        self.rewards = {
            s: {a: {s_next: 0.0 for s_next in self.S} for a in self.A}
            for s in self.S
        }
        self.transitions = {
            s: {a: {s_next: 0.0 for s_next in self.S} for a in self.A}
            for s in self.S
        }
        self.q_values = {s: {a: 0.0 for a in self.A} for s in self.S}

    def get_action(self, state: Any) -> Any:
        q_values = list(self.q_values[state].values())
        return np.argmax(q_values).astype(int)

    def get_random_action(self) -> Any:
        return self.env.action_space.sample()

    def get_samples(self, num_episodes: int) -> None:
        pass

    def compute_q_values(self) -> None:
        pass

    def train(self, num_iterations: int, num_episodes: int) -> None:
        pass

    def play(self, num_episodes: int, render: bool = True) -> None:
        if render:
            _, ax = plt.subplots(figsize=(8, 8))
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0.0
            while True:
                action = self.get_action(state)
                if render:
                    print(f"Action: {action_map(action)}")
                    plotting_fn(state, ax)
                state, reward, done, _ = self.env.step(action)
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
