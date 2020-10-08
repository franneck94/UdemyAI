import math

import gym
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Activation
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical


def reward_func(state, action):
    min_position = -1.2
    max_position = 0.6
    max_speed = 0.07
    goal_position = 0.5
    position, velocity = state
    velocity += (action - 1) * 0.001 + math.cos(3 * position) * (-0.0025)
    velocity = np.clip(velocity, -max_speed, max_speed)
    position += velocity
    position = np.clip(position, min_position, max_position)
    if (position == min_position and velocity < 0):
        velocity = 0

    done = bool(position >= goal_position)
    reward = abs(velocity)

    if done:
        reward += 100
    return reward


class Agent:
    # Constructor: Env, NN, Obs, Action
    def __init__(self, env):
        self.env = env
        self.observations = self.env.observation_space.shape[0]
        self.actions = self.env.action_space.n
        self.model = self.get_model()

    # Keras NN Model
    def get_model(self):
        model = Sequential()
        model.add(Dense(100, input_dim=self.observations)) # Input: State s
        model.add(Activation("relu"))
        model.add(Dense(self.actions)) # Output: Action [L, R]
        model.add(Activation("softmax"))
        model.summary()
        model.compile(
            optimizer=Adam(lr=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model

    # Based on the state/observation, get the action
    def get_action(self, observation):
        # get an action from our policy
        observation = observation.reshape(1, -1)
        action = self.model.predict(observation)[0] # [0.9, 0.1]
        action = np.random.choice(self.actions, p=action) # [L=0, R=1], p[0.9, 0.1]
        return action

    # Sample "random" games
    def get_samples(self, num_episodes):
        rewards = [0.0 for i in range(num_episodes)]
        episodes = [[] for i in range(num_episodes)]

        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0.0

            while True:
                action = self.get_action(state)
                new_state, _, done, _ = self.env.step(action)
                reward = reward_func(state, action)
                total_reward += reward
                episodes[episode].append((state, action))
                state = new_state
                if done:
                    rewards[episode] = total_reward
                    break

        return rewards, episodes

    # Helper function for the train function
    def filter_episodes(self, rewards, episodes, percentile):
        # [1, 2, 3, 4, 5, 6, ,7 ,8, 9, 10]
        # bound = 7
        # keep: 7 ,8, 9, 10
        reward_bound = np.percentile(rewards, percentile)
        x_train, y_train = [], []
        for r, e in zip(rewards, episodes):
            if r >= reward_bound:
                obs = [step[0] for step in e]
                acts = [step[1] for step in e]
                x_train.extend(obs)
                y_train.extend(acts)
        x_train = np.asarray(x_train)
        y_train = to_categorical(y_train, num_classes=self.actions) # L=0 => [1, 0]
        return x_train, y_train, reward_bound

    # Sample random games and train the NN
    def train(self, percentile, num_iterations, num_episodes):
        reward_means, reward_bounds = [], []
        for _ in range(num_iterations):
            rewards, episodes = self.get_samples(num_episodes=num_episodes)
            x_train, y_train, reward_bound = self.filter_episodes(rewards, episodes, percentile)
            self.model.fit(x_train, y_train)
            reward_mean = np.mean(rewards)
            print("Rewards Mean: ", reward_mean, " - Rewards Bound: ", reward_bound)
            reward_bounds.append(reward_bound)
            reward_means.append(reward_mean)
            if reward_mean > 500:
                break
        self.model.save("C:/Users/Jan/Dropbox/_Programmieren/UdemyAI/data/NN_mountain.hd5")
        return reward_means, reward_bounds

    # "Testing" of the Agent
    def play(self, num_episodes, render=True):
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0.0
            while True:
                if render:
                    self.env.render()
                action = self.get_action(state)
                state, _, done, _ = self.env.step(action)
                reward = reward_func(state, action)
                total_reward += reward
                if done:
                    print("Episode: ", episode, " - Reward: ", total_reward)
                    break


if __name__ == "__main__":
    env = gym.make("MountainCar-v0")
    agent = Agent(env)
    reward_means, reward_bounds = agent.train(percentile=90.0, num_iterations=50, num_episodes=100)
    input("Weiter?")
    agent.play(num_episodes=10, render=True)

    plt.plot(range(len(reward_means)), reward_means, color="red")
    plt.plot(range(len(reward_bounds)), reward_bounds, color="blue")
    plt.show()
