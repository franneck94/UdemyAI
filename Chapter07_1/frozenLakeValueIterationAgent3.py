from typing import Any

import gym
import matplotlib.pyplot as plt
import numpy as np

from plotting import action_map
from plotting import plotting_fn


class Agent:
    def __init__(self, env: gym.Env) -> None:
        self.env = env
        self.observations: int = self.env.observation_space.n
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
        for _ in range(num_episodes):
            action = self.get_random_action()
            new_state, reward, done, _ = self.env.step(action)
            self.rewards[self.state][action][new_state] = reward
            self.transitions[self.state][action][new_state] += 1
            if done:
                self.state = self.env.reset()
            else:
                self.state = new_state

    def compute_q_values(self) -> None:
        for s in self.S:
            for a in self.A:
                q_value = 0.0
                transitions_dict = self.transitions[s][a]
                transisions_list = list(transitions_dict.values())
                total_transitions = np.sum(transisions_list).astype(int)
                if total_transitions > 0:
                    for s_next, count in transitions_dict.items():
                        reward = self.rewards[s][a][s_next]
                        best_action = self.get_action(s_next)
                        q_value += (count / total_transitions) * (
                            reward
                            + self.gamma * self.q_values[s_next][best_action]
                        )
                    self.q_values[s][a] = q_value

    def train(self, num_iterations: int, num_episodes: int) -> None:
        self.get_samples(num_episodes=1_000)
        for _ in range(num_iterations):
            self.get_samples(num_episodes=num_episodes)
            self.compute_q_values()
            reward_mean = self.play(num_episodes=20, render=False)
            if reward_mean >= 0.9:
                break

    def play(self, num_episodes: int, render: bool = True) -> float:
        reward_sum = 0.0
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
                    reward_sum += total_reward
                    break
            print(f"Episode: {episode} Total Reward: {total_reward}")
        self.env.close()
        return reward_sum / num_episodes


def main() -> None:
    env = gym.make("FrozenLake-v0")
    agent = Agent(env)
    agent.train(num_iterations=10_000, num_episodes=1_000)
    agent.play(num_episodes=5)


if __name__ == "__main__":
    main()
