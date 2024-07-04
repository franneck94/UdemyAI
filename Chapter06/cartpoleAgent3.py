# type: ignore
from typing import Any

import gym
import matplotlib.pyplot as plt  # noqa: F401, RUF100
import numpy as np
from keras.layers import Activation
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical  # noqa: F401, RUF100


class Agent:
    def __init__(self, env: gym.Env) -> None:
        self.env = env
        self.observations: int = self.env.observation_space.shape[0]
        self.actions: int = self.env.action_space.n
        self.model = self.get_model()

    def get_model(self) -> Sequential:
        model = Sequential()
        model.add(Dense(units=100, input_dim=self.observations))
        model.add(Activation("relu"))
        model.add(Dense(units=self.actions))
        model.add(Activation("softmax"))
        model.summary()
        model.compile(
            optimizer=Adam(learning_rate=0.007),
            loss="categorical_crossentropy",
        )
        return model

    def get_action(self, state: Any) -> Any:
        return self.env.action_space.sample()

    def get_samples(self, num_episodes: int) -> tuple[list[float], list[float]]:
        rewards = [0.0 for _ in range(num_episodes)]
        epsiodes: list[Any] = [[] for _ in range(num_episodes)]

        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0.0

            while True:
                action = self.get_action(state)
                new_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                epsiodes[episode].append((state, action))
                state = new_state
                if done:
                    rewards[episode] = total_reward
                    break

        return rewards, epsiodes

    def filter_episodes(
        self,
        rewards: list[float],
        episodes: list[tuple[float, float]],
        percentile: float,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        pass

    def train(
        self,
        percentile: float,
        num_iterations: int,
        num_episodes: int,
    ) -> tuple[list[float], list[float]]:
        reward_means: list[float] = []
        reward_bounds: list[float] = []
        for it in range(num_iterations):
            rewards, episodes = self.get_samples(num_episodes)
            x_train, y_train, reward_bound = self.filter_episodes(
                rewards,
                episodes,
                percentile,
            )
            self.model.train_on_batch(x=x_train, y=y_train)
            reward_mean = np.mean(rewards)
            print(
                f"Iteration: {it:2d} "
                f"Reward Mean: {reward_mean:.4f} "
                f"Reward Bound: {reward_bound:.4f}",
            )
            reward_bounds.append(reward_bound)
            reward_means.append(reward_mean)
        return reward_means, reward_bounds

    def play(self, episodes: int, render: bool = True) -> None:
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
        self.env.close()


def main() -> None:
    env = gym.make("CartPole-v1")
    agent = Agent(env)
    agent.train(percentile=70.0, num_iterations=30, num_episodes=100)


if __name__ == "__main__":
    main()
