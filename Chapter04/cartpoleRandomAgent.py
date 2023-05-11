from typing import Any

import gym
import numpy as np


class Agent:
    def __init__(self, env: gym.Env) -> None:
        self.env = env

    def get_action(self) -> Any:
        return self.env.action_space.sample()

    def play(self, episodes: int, render: bool = True) -> list:
        rewards = [0.0 for _ in range(episodes)]
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
            rewards.append(total_reward)
        self.env.close()
        return rewards


def main() -> None:
    env = gym.make("CartPole-v1")
    agent = Agent(env)
    rewards = agent.play(episodes=100, render=False)

    rewards_mean = np.mean(rewards)
    rewards_min = np.min(rewards)
    rewards_max = np.max(rewards)

    print(f"rewards_mean: {rewards_mean}")
    print(f"rewards_min: {rewards_min}")
    print(f"rewards_max: {rewards_max}")


if __name__ == "__main__":
    main()
