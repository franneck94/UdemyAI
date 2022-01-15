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
        action = np.argmax(q_values).astype(int)
        return action

    def get_random_action(self) -> Any:
        action = self.env.action_space.sample()
        return action

    def get_v_values(self, state: Any) -> float:
        q_values = list(self.q_values[state].values())
        v_value: float = np.max(q_values).astype(float)
        return v_value

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
        self, state: Any, action: Any, reward: float, state_next: Any
    ) -> None:
        v_value_next = self.get_v_values(state_next)
        update_q_value = reward + self.gamma * v_value_next
        q_value_action = self.q_values[state][action]
        new_q_value = (
            1.0 - self.alpha
        ) * q_value_action + self.alpha * update_q_value
        self.q_values[state][action] = new_q_value

    def train(self, num_iterations: int) -> None:
        best_reward_mean = -np.inf
        for iteration in range(num_iterations):
            state, action, reward, next_state = self.get_sample()
            self.compute_q_values(state, action, reward, next_state)
            reward_mean = self.play(num_episodes=20, render=False)
            if iteration % 250 == 0:
                print(f"Iteration: {iteration}")
            if reward_mean > best_reward_mean:
                print(
                    f"Old best_reward_mean: {best_reward_mean}",
                    f"New best_reward_mean: {reward_mean}",
                )
                best_reward_mean = reward_mean

    def play(self, num_episodes: int, render: bool = True) -> float:
        env = gym.make("FrozenLake-v1")
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
            if render:
                print(f"Episode: {episode} Total Reward: {total_reward}")
        env.close()
        return reward_sum / num_episodes


def main() -> None:
    env = gym.make("FrozenLake-v1")
    agent = Agent(env)
    agent.train(num_iterations=20_000)
    agent.play(num_episodes=5, render=True)
    save_map(agent.q_values, name="tabularq.png")


if __name__ == "__main__":
    main()
