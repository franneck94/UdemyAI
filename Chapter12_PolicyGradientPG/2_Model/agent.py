import random
import collections

import gym
import numpy as np
import matplotlib.pyplot as plt
    
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.utils import *

from dqn import *

class Agent:
    def __init__(self, env):
        # DQN Env Variables
        self.env = env
        self.num_observations = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n
        self.gamma = 0.99
        self.learning_rate = 1e-3
        self.model = DQN(self.num_observations, self.num_actions, self.learning_rate)

    def get_action(self, state):
        pass

    def train(self, num_episodes):

        for episode in range(num_episodes):
            
            while True:
                
                if done:

    def play(self, num_episodes, render=True):
        
        for episode in range(num_episodes):

            while True:

                if render:
                    self.env.render()
                if done:
                    break

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = Agent(env)
    agent.train(num_episodes=1000)
    input("Play?")
    agent.play(num_episodes=15, render=True)