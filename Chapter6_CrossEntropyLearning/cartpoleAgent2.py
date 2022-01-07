from typing import Any
from typing import List
from typing import Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


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
            optimizer=Adam(learning_rate=0.007), loss="categorical_crossentropy"
        )
        return model

    def get_action(self) -> Any:
        return self.env.action_space.sample()

    def get_samples(self, num_episodes: int) -> Tuple[List[float], List[float]]:
        pass

    def filter_episodes(
        self,
        rewards: List[float],
        episodes: List[Tuple[float, float]],
        percentile: float,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        pass

    def train(
        self, percentile: float, num_iterations: int, num_episodes: int
    ) -> Tuple[List[float], List[float]]:
        pass

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


if __name__ == "__main__":
    main()
