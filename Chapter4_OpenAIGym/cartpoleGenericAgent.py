from typing import Any

import gym


class Agent:
    def __init__(self, env: gym.Env) -> None:
        self.env = env

    def get_action(self) -> Any:
        action = self.env.action_space.sample()
        return action

    def play(self, episodes: int, render: bool = True) -> list:
        rewards = [0.0 for i in range(episodes)]

        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0.0

            while True:
                if render:
                    self.env.render()
                action = self.get_action()
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
                if done:
                    rewards[episode] = total_reward
                    break

        return rewards


def main() -> None:
    env = gym.make("CartPole-v1")
    agent = Agent(env)
    _ = agent.play(episodes=100, render=True)


if __name__ == "__main__":
    main()
