import collections

import gym
import matplotlib.pyplot as plt
import numpy as np

from plotting import *


class Agent:
    def __init__(self, env):
        self.env = env
        self.observations = self.env.observation_space.n
        self.actions = self.env.action_space.n
        self.S = range(self.observations)
        self.A = range(self.actions)
        self.gamma = 0.95
        self.alpha = 0.25
        self.values = {s: {a: 0.0 for a in self.A} for s in self.S}
        self.state = self.env.reset()

    def get_action(self, s_next):
        act = np.argmax(list(self.values[s_next].values()))
        return act

    def get_value(self, s_next):
        act = np.max(list(self.values[s_next].values()))
        return act

    def get_random_action(self):
        action = self.env.action_space.sample()
        return action

    def get_sample(self):
        old_state = self.state
        action = self.get_random_action()
        new_state, reward, done, _ = self.env.step(action)
        if done:
            self.state = self.env.reset()
        else:
            self.state = new_state
        return (old_state, action, reward, new_state)

    def update_q_values(self, s, a, r, s_next):
        # Q(s', a') = q_prime
        q_prime = self.get_value(s_next)
        update_q = r + self.gamma * q_prime
        q = self.values[s][a]
        # Q(s,a) = (1 - allpha) * Q(s,a) + alpha * (r + gamma * max(a') Q(s',a'))
        q = (1 - self.alpha) * q + self.alpha * update_q
        self.values[s][a] = q

    def train(self, num_iterations):
        for iteration in range(num_iterations):
            s, a, r, s_next = self.get_sample()
            self.update_q_values(s, a, r, s_next)
            mean_reward = self.test(num_episodes=20)
            if mean_reward >= 0.9:
                break

    def test(self, num_episodes):
        env = gym.make("FrozenLake-v0")
        sum_rewards = 0.0
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0.0
            while True:
                action = self.get_action(state)
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
                if done:
                    sum_rewards += total_reward
                    break
        return sum_rewards / num_episodes

    def play(self, num_episodes, render=True):
        fig, ax = plt.subplots()
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0.0
            while True:
                if render:
                    plotting_fn(state, ax)
                action = self.get_action(state)
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
                if done:
                    print("Episode: ", episode, " - Reward: ", total_reward)
                    break


if __name__ == "__main__":
    env = gym.make("FrozenLake-v0")
    agent = Agent(env)
    agent.train(num_iterations=10000)
    agent.play(num_episodes=10)
