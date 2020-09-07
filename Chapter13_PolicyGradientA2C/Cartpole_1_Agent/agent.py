import gym
import numpy as np
import matplotlib.pyplot as plt

from nn import NN


class Agent:
    def __init__(self, env):
        self.env = env
        self.num_observations = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n
        self.num_values = 1
        self.gamma = 0.99
        self.lr_actor = 1e-4
        self.lr_critic = 1e-4
        self.model = NN(self.num_observations, self.num_actions, self.num_values, self.lr_actor, self.lr_critic)

    def get_action(self, state):
        pass

    def update_policy(self):
        pass

    def train(self, num_episodes):
        total_rewards = []
        for episode in range(num_episodes):

            while True:

                if done:
                    break

    def play(self, num_episodes):
        for episode in range(num_episodes):

            while True:

                if done:
                    break


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = Agent(env)
    total_rewards = agent.train(num_episodes=1000)
