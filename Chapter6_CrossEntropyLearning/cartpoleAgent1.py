import gym
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.utils import *


class Agent:
    # Constructor: Env, NN, Obs, Action
    def __init__(self, env):
        pass

    # Keras NN Model
    def get_model(self):
        pass

    # Based on the state/observation, get the action
    def get_action(self):
        pass

    # Sample "random" games
    def get_samples(self):
        pass

    # Helper function for the train function
    def filter_episodes(self):
        pass

    # Sample random games and train the NN
    def train(self):
        pass

    # "Testing" of the Agent
    def play(self):
        pass


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = Agent(env)
    agent.train()
    agent.play()
