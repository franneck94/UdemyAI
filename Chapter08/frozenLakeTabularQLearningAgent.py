from typing import Any

import gym
import matplotlib.pyplot as plt
import numpy as np

from plotting import action_map
from plotting import plotting_q_values
from plotting import save_map


class Agent:
    def __init__(self, env: gym.Env) -> None:
        self.env = env
        self.observations: int = self.env.observation_space.n
        self.actions: int = self.env.action_space.n
        self.gamma = 0.95
        self.alpha = 0.20
        self.state = self.env.reset()
        self.S = range(self.observations)
        self.A = range(self.actions)
        self.q_values = {s: {a: 0.0 for a in self.A} for s in self.S}

    def get_action(self, state: Any) -> Any:
        q_values = list(self.q_values[state].values())
        return np.argmax(q_values).astype(int)

    def get_random_action(self) -> Any:
        return self.env.action_space.sample()

    def get_sample(self) -> None:
        action = self.get_random_action()
        new_state, _, done, _ = self.env.step(action)
        if done:
            self.state = self.env.reset()
        else:
            self.state = new_state

    def compute_q_values(self) -> None:
        pass

    def train(self, num_iterations: int) -> None:
        pass

    def play(self, num_episodes: int, render: bool = True) -> float:
        env = gym.make("FrozenLake-v0")
        reward_sum = 0.0
        if render:
            _, ax = plt.subplots(figsize=(8, 8))
        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0.0
            while True:
                action = self.get_action(state)
                if render:
                    print(f"Action: {action_map(action)}")
                    plotting_q_values(state, action, self.q_values, ax)
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    reward_sum += total_reward
                    break
            print(f"Episode: {episode} Total Reward: {total_reward}")
        env.close()
        return reward_sum / num_episodes


def main() -> None:
    env = gym.make("FrozenLake-v0")
    agent = Agent(env)
    agent.train(num_iterations=20_000)
    agent.play(num_episodes=5, render=True)
    save_map(agent.q_values, name="tabularq.png")


if __name__ == "__main__":
    main()
