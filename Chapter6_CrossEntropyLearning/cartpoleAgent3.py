from typing import Any

import gym
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


class Agent:
    """Agent class for the cross-entropy learning algorithm."""

    def __init__(self, env: gym.Env) -> None:
        """Set up the environment, the neural network and member variables.

        Parameters
        ----------
        env : gym.Environment
            The game environment
        """
        self.env = env
        self.observations = self.env.observation_space.shape[0]
        self.actions = self.env.action_space.n
        self.model = self.get_model()

    def get_model(self) -> Sequential:
        """Returns a keras NN model."""
        model = Sequential()
        model.add(Dense(units=100, input_dim=self.observations))
        model.add(Activation("relu"))
        model.add(Dense(units=self.actions))  # Output: Action [L, R]
        model.add(Activation("softmax"))
        model.summary()
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def get_action(self, state: np.ndarray) -> Any:
        """Based on the state, get an action."""
        state = state.reshape(1, -1)  # [4,] => [1, 4]
        action = self.model(state).numpy()[0]
        action = np.random.choice(
            self.actions, p=action
        )  # choice([0, 1], [0.5044534  0.49554658])
        return action

    def get_samples(self):
        """Sample games."""
        pass

    def filter_episodes(self):
        """Helper function for the training."""
        pass

    def train(self):
        """Play games and train the NN."""
        pass

    def play(self, num_episodes: int, render: bool = True) -> None:
        """Test the trained agent."""
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0.0
            while True:
                if render:
                    self.env.render()
                action = self.get_action(state)
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
                if done:
                    print(
                        f"Total reward: {total_reward} in epsiode {episode + 1}"
                    )
                    break


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = Agent(env)
    print(agent.observations)
    print(agent.actions)
    agent.train()
    agent.play(num_episodes=10)
