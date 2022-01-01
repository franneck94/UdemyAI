from typing import Any

import gym
import matplotlib.pyplot as plt
import numpy as np

from plotting import plotting_fn


class Agent:
    def __init__(self, env: gym.Env) -> None:
        self.env = env
        self.observations = self.env.observation_space.n
        self.actions = self.env.action_space.n
        self.S = range(self.observations)
        self.A = range(self.actions)
        self.gamma = 0.9
        self.rewards = {
            s: {a: {s_next: 0 for s_next in self.S} for a in self.A}
            for s in self.S
        }
        self.transitions = {
            s: {a: {s_next: 0 for s_next in self.S} for a in self.A}
            for s in self.S
        }
        self.values = {s: {a: 0.0 for a in self.A} for s in self.S}

    def get_action(self, s_next: int) -> Any:
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

    def test(self, num_episodes: int) -> float:
        pass

    def play(self, num_episodes: int, render: bool = True) -> None:
        fig, ax = plt.subplots()
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0.0
            while True:
                if render:
                    plotting_fn(state, ax)
                action = self.get_action(state)
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
                if done:
                    print("Episode: ", episode, " - Reward: ", total_reward)
                    break


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1")
    agent = Agent(env)
    agent.train(num_iterations=50, num_episodes=100)
    agent.play(num_episodes=20)
