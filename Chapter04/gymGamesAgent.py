from typing import Any

import gym
import numpy as np


class Agent:
    def __init__(self, env: gym.Env) -> None:
        self.env = env

    def get_action(self) -> Any:
        return self.env.action_space.sample()

    def play(self, episodes: int, render: bool = True) -> list:
        rewards = [0.0 for i in range(episodes)]

        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0.0

            while True:
                if render:
                    self.env.render()
                action = self.get_action()
                state, reward, done, _ = self.env.step(action)  # noqa: F841
                total_reward += reward
                if done:
                    rewards[episode] = total_reward
                    break

        return rewards


def main() -> None:
    games = [
        "CartPole-v1",
        "MountainCar-v0",
        "PongNoFrameskip-v4",
        "Breakout-v0",
    ]

    for game in games:
        env = gym.make(game)
        agent = Agent(env)
        rewards = agent.play(episodes=100, render=True)

        rewards_mean = np.mean(rewards)
        rewards_min = np.min(rewards)
        rewards_max = np.max(rewards)

        print("Rewards Mean: ", rewards_mean)
        print("Rewards Min: ", rewards_min)
        print("Rewards Max: ", rewards_max)


if __name__ == "__main__":
    main()
