import gym
import numpy as np
import matplotlib.pyplot as plt

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.utils import *


class Agent:
    # Constructor: Env, NN, Obs, Action
    def __init__(self, env):
        self.env = env
        self.observations = self.env.observation_space.shape[0]
        self.actions = self.env.action_space.n
        self.model = self.get_model()

    # Keras NN Model
    def get_model(self):
        m = Sequential()
        m.add(Dense(100, input_dim=self.observations)) # Input: State s
        m.add(Activation("relu"))
        m.add(Dense(self.actions)) # Output: Action [L, R]
        m.add(Activation("softmax"))
        m.summary()
        m.compile(optimizer=Adam(lr=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
        return m

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
                new_state, reward, done, _ = self.env.step(action)
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
        for iteration in range(num_iterations):
            rewards, episodes = self.get_samples(num_episodes=num_episodes)
            x_train, y_train, reward_bound = self.filter_episodes(rewards, episodes, percentile)
            self.model.fit(x_train, y_train)
            reward_mean = np.mean(rewards)
            print("Rewards Mean: ", reward_mean, " - Rewards Bound: ", reward_bound)
            reward_bounds.append(reward_bound)
            reward_means.append(reward_mean)
            if reward_mean > 495:
                break
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
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
                if done:
                    print("Episode: ", episode, " - Reward: ", total_reward)
                    break


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = Agent(env)
    reward_means, reward_bounds = agent.train(percentile=70.0, num_iterations=25, num_episodes=100)
    # agent.play(num_episodes=3, render=True)

    plt.plot(range(len(reward_means)), reward_means, color="red")
    plt.savefig('./agent CE.png')
