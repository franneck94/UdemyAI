import gym
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


class Agent:
    """Agent class for the cross-entropy learning algorithm.
    """

    def __init__(self, env):
        """Set up the environment, the neural network and member variables.

        Parameters
        ----------
        env : gym.Environment
            The game environment
        """
        pass

    def get_model(self):
        """Keras NN Model.
        """
        pass

    def get_action(self):
        """Based on the state, get an action.
        """
        pass

    def get_samples(self):
        """Sample games.
        """
        pass

    def filter_episodes(self):
        """Helper function for the training.
        """
        pass

    def train(self):
        """Play games and train the NN.
        """
        pass

    def play(self):
        """Test the trained agent.
        """
        pass


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = Agent(env)
    agent.train()
    agent.play()
