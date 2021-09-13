import math

import gym
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical


def reward_func(state, action):
    """Custom reward function for the mountain car game.

    Parameters
    ----------
    state : np.ndarray
        [description]
    action : int
        L = 0, R = 1

    Returns
    -------
    float
        Reward of the action
    """
    min_position = -1.2
    max_position = 0.6
    max_speed = 0.07
    goal_position = 0.5
    position, velocity = state
    velocity += (action - 1) * 0.001 + math.cos(3 * position) * (-0.0025)
    velocity = np.clip(velocity, -max_speed, max_speed)
    position += velocity
    position = np.clip(position, min_position, max_position)
    if position == min_position and velocity < 0:
        velocity = 0

    done = bool(position >= goal_position)
    reward = abs(velocity)

    if done:
        reward += 100.0
    return reward


class Agent:
    """Agent class for the cross-entropy learning algorithm."""

    def __init__(self, env):
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

    def get_model(self):
        """Returns a keras NN model."""
        model = Sequential()
        model.add(Dense(units=100, input_dim=self.observations))
        model.add(Activation("relu"))
        model.add(Dense(units=self.actions))  # Output: Action [L, R]
        model.add(Activation("softmax"))
        model.summary()
        model.compile(
            optimizer=RMSprop(),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def get_action(self, state: np.ndarray):
        """Based on the state, get an action."""
        state = state.reshape(1, -1)  # [4,] => [1, 4]
        action = self.model(state).numpy()[0]
        action = np.random.choice(
            self.actions, p=action
        )  # choice([0, 1], [0.5044534  0.49554658])
        return action

    def get_samples(self, num_episodes: int):
        """Sample games."""
        rewards = [0.0 for i in range(num_episodes)]
        episodes = [[] for i in range(num_episodes)]

        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0.0

            while True:
                action = self.get_action(state)
                new_state, reward, done, _ = self.env.step(action)
                reward = reward_func(state, action)
                total_reward += reward
                episodes[episode].append((state, action))
                state = new_state
                if done:
                    rewards[episode] = total_reward
                    break

        return rewards, episodes

    def filter_episodes(self, rewards, episodes, percentile):
        """Helper function for the training."""
        reward_bound = np.percentile(rewards, percentile)
        x_train, y_train = [], []
        for reward, episode in zip(rewards, episodes):
            if reward >= reward_bound:
                observation = [step[0] for step in episode]
                action = [step[1] for step in episode]
                x_train.extend(observation)
                y_train.extend(action)
        x_train = np.asarray(x_train)
        y_train = to_categorical(
            y_train, num_classes=self.actions
        )  # L = 0 => [1, 0]
        return x_train, y_train, reward_bound

    def train(self, percentile, num_iterations, num_episodes):
        """Play games and train the NN."""
        reward_means, reward_bounds = [], []
        for _ in range(num_iterations):
            rewards, episodes = self.get_samples(num_episodes)
            x_train, y_train, reward_bound = self.filter_episodes(
                rewards, episodes, percentile
            )
            self.model((x_train, y_train), training=True)
            reward_mean = np.mean(rewards)
            print(f"Reward mean: {reward_mean}, reward bound: {reward_bound}")
            reward_bounds.append(reward_bound)
            reward_means.append(reward_mean)
            if reward_mean > 500:
                break
        return reward_means, reward_bounds

    def play(self, num_episodes: int, render: bool = True):
        """Test the trained agent."""
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0.0
            while True:
                if render:
                    self.env.render()
                action = self.get_action(state)
                state, reward, done, _ = self.env.step(action)
                reward = reward_func(state, action)
                total_reward += reward
                if done:
                    print(
                        f"Total reward: {total_reward} in epsiode {episode + 1}"
                    )
                    break


if __name__ == "__main__":
    env = gym.make("MountainCar-v0")
    agent = Agent(env)

    reward_means, reward_bounds = agent.train(
        percentile=70.0, num_iterations=20, num_episodes=50
    )
    input("Weiter?")
    agent.play(num_episodes=10, render=True)

    plt.plot(range(len(reward_means)), reward_means, color="red")
    plt.plot(range(len(reward_bounds)), reward_bounds, color="blue")
    plt.show()
