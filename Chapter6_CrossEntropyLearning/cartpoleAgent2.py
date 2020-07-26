import gym
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import *
from keras.models import *
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
        # return keras model
        pass

    # Based on the state/observation, get the action
    def get_action(self):
        # sample action from "trained" policy
        pass

    # Sample "random" games
    def get_samples(self):
        # play some "random" games with our current policy
        pass

    # Helper function for the train function
    def filter_episodes(self):
        # filter an episode by the reward
        # x, y trainset for the NN
        pass

    # Sample random games and train the NN
    def train(self):
        # for iteration time:
        #     x,y = get_smaples()
        #     nn train(x, y)
        #     monitor performance
        pass

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
    agent.train()
    agent.play(num_episodes=10)