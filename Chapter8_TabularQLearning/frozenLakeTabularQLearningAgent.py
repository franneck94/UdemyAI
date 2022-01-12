from typing import Any

import gym
import matplotlib.pyplot as plt
import numpy as np

from plotting import plotting_q_values
from plotting import save_map
from plotting import action_map


class Agent:
    def __init__(self, env: gym.Env) -> None:
        self.env = env
        self.observations = self.env.observation_space.n
        self.actions = self.env.action_space.n
        self.S = range(self.observations)
        self.A = range(self.actions)
        self.gamma = 0.95
        self.alpha = 0.20
        self.q_values = {s: {a: 0.0 for a in self.A} for s in self.S}
        self.state = self.env.reset()

    def get_action(self, s_next: int) -> float:
        act: float = np.argmax(list(self.q_values[s_next].values()))
        return act

    def get_random_action(self) -> Any:
        action = self.env.action_space.sample()
        return action

    def get_value(self, s_next: Any) -> float:
        value: float = np.max(list(self.q_values[s_next].values()))
        return value

    def get_sample(self) -> tuple:
        old_state = self.state
        action = self.get_random_action()
        new_state, reward, done, _ = self.env.step(action)
        if done:
            self.state = self.env.reset()
        else:
            self.state = new_state
        return (old_state, action, reward, new_state)

    def compute_q_values(
        self, s: Any, a: Any, reward: float, s_next: Any
    ) -> None:
        q_values_next = self.get_value(s_next)
        update_q = reward + self.gamma * q_values_next
        q_value_action = self.q_values[s][a]
        q_value_action = (
            1 - self.alpha
        ) * q_value_action + self.alpha * update_q
        self.q_values[s][a] = q_value_action

    def train(self, num_iterations: int) -> None:
        best_reward_mean = -np.inf
        for iteration in range(num_iterations):
            s, a, reward, s_next = self.get_sample()
            self.compute_q_values(s, a, reward, s_next)
            reward_mean = self.play(num_episodes=20, render=False)
            if iteration % 250 == 0:
                print(f"Iteration: {iteration}")
            if reward_mean > best_reward_mean:
                print(
                    f" Old BestMeanReward: {best_reward_mean}",
                    f" New BestMeanReward: {reward_mean}",
                )
                best_reward_mean = reward_mean
            if reward_mean >= 0.9:
                break

    def play(self, num_episodes: int, render: bool = True) -> float:
        env = gym.make("FrozenLake-v1")
        if render:
            _, ax = plt.subplots(figsize=(8, 8))
        reward_sum = 0.0
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
            if render:
                print(f"Episode: {episode} Total Reward: {total_reward}")
        self.env.close()
        return reward_sum / num_episodes


def main() -> None:
    env = gym.make("FrozenLake-v1")
    agent = Agent(env)
    agent.train(num_iterations=20_000)
    agent.play(num_episodes=3, render=True)
    save_map(agent.q_values, name="tabularq.png")


if __name__ == "__main__":
    main()
